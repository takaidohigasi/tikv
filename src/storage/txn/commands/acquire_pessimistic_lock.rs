// Copyright 2020 TiKV Project Authors. Licensed under Apache-2.0.

// #[PerformanceCriticalPath]
use kvproto::kvrpcpb::ExtraOp;
use resource_metering::record_network_out_bytes;
use tikv_kv::Modify;
use txn_types::{Key, OldValues, TimeStamp, TxnExtra, insert_old_value_if_resolved};

use crate::storage::{
    Error as StorageError, PessimisticLockKeyResult, ProcessResult, Result as StorageResult,
    Snapshot,
    kv::WriteData,
    lock_manager::{LockManager, WaitTimeout},
    mvcc::{Error as MvccError, ErrorInner as MvccErrorInner, MvccTxn, SnapshotReader},
    txn::{
        Error, ErrorInner, Result, acquire_pessimistic_lock,
        commands::{
            Command, CommandExt, ReaderWithStats, ReleasedLocks, ResponsePolicy, TypedCommand,
            WriteCommand, WriteContext, WriteResult, WriteResultLockInfo,
        },
    },
    types::{PessimisticLockParameters, PessimisticLockResults},
};

command! {
    /// Acquire a Pessimistic lock on the keys.
    ///
    /// This can be rolled back with a [`PessimisticRollback`](Command::PessimisticRollback) command.
    AcquirePessimisticLock:
        cmd_ty => StorageResult<PessimisticLockResults>,
        display => {
            "kv::command::acquirepessimisticlock keys({:?}) @ {} {} {} {:?} {} {} {} | {:?}",
            (keys, start_ts, lock_ttl, for_update_ts, wait_timeout, min_commit_ts,
                check_existence, lock_only_if_exists, ctx),
        }
        content => {
            /// The set of keys to lock.
            /// (Key, bool, bool) means (key, should_not_exist, is_shared_lock)
            keys: Vec<(Key, bool, bool)>,
            /// The primary lock. Secondary locks (from `keys`) will refer to the primary lock.
            primary: Vec<u8>,
            /// The transaction timestamp.
            start_ts: TimeStamp,
            /// The Time To Live of the lock, in milliseconds
            lock_ttl: u64,
            is_first_lock: bool,
            for_update_ts: TimeStamp,
            /// Time to wait for lock released in milliseconds when encountering locks.
            wait_timeout: Option<WaitTimeout>,
            /// If it is true, TiKV will return values of the keys if no error, so TiDB can cache the values for
            /// later read in the same transaction.
            return_values: bool,
            min_commit_ts: TimeStamp,
            check_existence: bool,
            lock_only_if_exists: bool,
            allow_lock_with_conflict: bool,
            /// If set, keys locked by other transactions are skipped instead of waited for
            /// or reported as errors: no lock is written for them and each is reported as
            /// `PessimisticLockKeyResult::Skipped`. Implements `SELECT ... FOR UPDATE SKIP
            /// LOCKED`. Incompatible with `allow_lock_with_conflict`.
            skip_locked: bool,
        }
        in_heap => {
            primary,
            keys,
        }
}

impl CommandExt for AcquirePessimisticLock {
    ctx!();
    tag!(acquire_pessimistic_lock);
    request_type!(KvPessimisticLock);
    ts!(start_ts);
    property!(can_be_pipelined);

    fn write_bytes(&self) -> usize {
        self.keys
            .iter()
            .map(|(key, ..)| key.as_encoded().len())
            .sum()
    }

    gen_lock!(keys: multiple(|x| &x.0));
}

impl<S: Snapshot, L: LockManager> WriteCommand<S, L> for AcquirePessimisticLock {
    fn process_write(self, snapshot: S, context: WriteContext<'_, L>) -> Result<WriteResult> {
        if self.allow_lock_with_conflict && self.keys.len() > 1 {
            // Currently multiple keys with `allow_lock_with_conflict` set is not supported.
            return Err(Error::from(ErrorInner::Other(box_err!(
                "multiple keys in a single request with allowed_lock_with_conflict set is not allowed"
            ))));
        }
        if self.skip_locked && self.allow_lock_with_conflict {
            // `skip_locked` skips keys locked by other transactions while
            // `allow_lock_with_conflict` forces locking through write conflicts; combining
            // them is not defined.
            return Err(Error::from(ErrorInner::Other(box_err!(
                "skip_locked together with allow_lock_with_conflict is not allowed"
            ))));
        }

        let (start_ts, ctx, keys) = (self.start_ts, self.ctx, self.keys);
        let mut txn = MvccTxn::new(start_ts, context.concurrency_manager);
        let mut reader = ReaderWithStats::new(
            SnapshotReader::new_with_ctx(start_ts, snapshot, &ctx),
            context.statistics,
        );

        let total_keys = keys.len();
        let mut res = PessimisticLockResults::with_capacity(total_keys);
        let mut encountered_locks = vec![];
        let mut updated_shared_lock_info = None;
        let need_old_value = context.extra_op == ExtraOp::ReadOldValue;
        let mut old_values = OldValues::default();
        for (k, should_not_exist, is_shared_lock) in keys {
            match acquire_pessimistic_lock(
                &mut txn,
                &mut reader,
                k.clone(),
                &self.primary,
                should_not_exist,
                self.lock_ttl,
                self.for_update_ts,
                self.return_values,
                self.check_existence,
                self.min_commit_ts,
                need_old_value,
                self.lock_only_if_exists,
                self.allow_lock_with_conflict,
                is_shared_lock,
            ) {
                Ok((key_res, old_value)) => {
                    res.push(key_res);
                    // MutationType is unknown in AcquirePessimisticLock stage.
                    insert_old_value_if_resolved(&mut old_values, k, txn.start_ts, old_value, None);
                }
                Err(MvccError(box MvccErrorInner::KeyIsLocked(lock_info))) => {
                    if self.skip_locked {
                        // The key is locked by another transaction: skip it and keep locking
                        // the remaining keys. Previously succeeded keys stay locked, and the
                        // key never enters the lock waiting queue.
                        res.push(PessimisticLockKeyResult::Skipped);
                        continue;
                    }
                    let request_parameters = PessimisticLockParameters {
                        pb_ctx: ctx.clone(),
                        primary: self.primary.clone(),
                        start_ts,
                        lock_ttl: self.lock_ttl,
                        for_update_ts: self.for_update_ts,
                        wait_timeout: self.wait_timeout,
                        return_values: self.return_values,
                        min_commit_ts: self.min_commit_ts,
                        check_existence: self.check_existence,
                        is_first_lock: self.is_first_lock,
                        lock_only_if_exists: self.lock_only_if_exists,
                        allow_lock_with_conflict: self.allow_lock_with_conflict,
                    };
                    let lock_info = WriteResultLockInfo::new(
                        lock_info,
                        request_parameters,
                        k,
                        should_not_exist,
                        is_shared_lock,
                    );
                    encountered_locks.push(lock_info);
                    // Do not lock previously succeeded keys.
                    txn.clear();
                    res.0.clear();
                    res.push(PessimisticLockKeyResult::Waiting);
                    break;
                }
                Err(MvccError(box MvccErrorInner::NotInShrinkMode(mut shared_locks))) => {
                    if self.skip_locked {
                        // The key is share-locked by other transactions: skip it without
                        // converting the shared locks to shrink-only, since a skip-locked
                        // reader must not mutate other transactions' lock state.
                        res.push(PessimisticLockKeyResult::Skipped);
                        continue;
                    }
                    // Clear previous mutations, mark `shared_locks` as shrink-only and write it
                    // back.
                    let locked_raw_key = k.to_raw()?;
                    shared_locks.set_shrink_only();
                    txn.clear();
                    txn.put_shared_locks(k, &shared_locks, false);
                    old_values.clear();
                    res.0.clear();

                    updated_shared_lock_info = Some(shared_locks.into_lock_info(locked_raw_key));
                    break;
                }
                Err(e) => return Err(Error::from(e)),
            }
        }

        let new_acquired_locks = txn.take_new_locks();
        let modifies = txn.into_modifies();

        let mut res = Ok(res);

        // If encountered lock and `wait_timeout` is `None` (which means no wait),
        // return error directly here.
        if !encountered_locks.is_empty() && self.wait_timeout.is_none() {
            // Mind the difference of the protocols of legacy requests and resumable
            // requests. For resumable requests (allow_lock_with_conflict ==
            // true), key errors are considered key by key instead of for the
            // whole request.
            let lock_info = encountered_locks.drain(..).next().unwrap().lock_info_pb;
            let err = StorageError::from(Error::from(MvccError::from(
                MvccErrorInner::KeyIsLocked(lock_info),
            )));
            if self.allow_lock_with_conflict {
                res.as_mut().unwrap().0[0] = PessimisticLockKeyResult::Failed(err.into())
            } else {
                res = Err(err)
            }
        }
        // Return KeyIsLocked when xlock encounters slock and tries to convert it to
        // shrink-only.
        if let Some(lock_info) = updated_shared_lock_info {
            res = Err(StorageError::from(Error::from(MvccError::from(
                MvccErrorInner::KeyIsLocked(lock_info),
            ))));
        }

        let rows = if res.is_ok() { total_keys } else { 0 };

        record_network_out_bytes(res.as_ref().map_or(0, |v| v.estimate_resp_size()));
        let pr = ProcessResult::PessimisticLockRes { res };

        let to_be_write = make_write_data(modifies, old_values);

        Ok(WriteResult {
            ctx,
            to_be_write,
            rows,
            pr,
            lock_info: encountered_locks,
            released_locks: ReleasedLocks::new(),
            new_acquired_locks,
            lock_guards: vec![],
            response_policy: ResponsePolicy::OnProposed,
            known_txn_status: vec![],
        })
    }
}

pub(super) fn make_write_data(modifies: Vec<Modify>, old_values: OldValues) -> WriteData {
    if !modifies.is_empty() {
        let extra = TxnExtra {
            old_values,
            // One pc status is unknown in AcquirePessimisticLock stage.
            one_pc: false,
            allowed_in_flashback: false,
        };
        WriteData::new(modifies, extra)
    } else {
        WriteData::default()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use concurrency_manager::ConcurrencyManager;
    use kvproto::kvrpcpb;
    use txn_types::LockInfoExt;

    use super::*;
    use crate::storage::{
        Engine, Statistics, TestEngineBuilder,
        lock_manager::MockLockManager,
        mvcc::tests::must_load_shared_lock,
        txn::{
            actions::{
                acquire_pessimistic_lock::tests::must_pessimistic_locked,
                tests::{must_prewrite_delete, must_prewrite_lock, must_prewrite_put},
            },
            commands::test_util::pessimistic_lock,
            tests::{must_commit, must_rollback},
            txn_status_cache::TxnStatusCache,
        },
    };

    impl PartialEq for PessimisticLockKeyResult {
        fn eq(&self, other: &Self) -> bool {
            match (self, other) {
                (Self::Empty, Self::Empty) => true,
                (Self::Value(a), Self::Value(b)) => a == b,
                (Self::Existence(a), Self::Existence(b)) => a == b,
                (
                    Self::LockedWithConflict {
                        value: v1,
                        conflict_ts: ts1,
                    },
                    Self::LockedWithConflict {
                        value: v2,
                        conflict_ts: ts2,
                    },
                ) => v1 == v2 && ts1 == ts2,
                (Self::Waiting, Self::Waiting) => true,
                (Self::Skipped, Self::Skipped) => true,
                (Self::Failed(a), Self::Failed(b)) => format!("{:?}", a) == format!("{:?}", b),
                _ => false,
            }
        }
    }

    #[test]
    fn test_pessimistic_with_return_values() {
        let mut engine = TestEngineBuilder::new().build().unwrap();
        let mut statistics = Statistics::default();

        let k1 = b"k1";
        let k2 = b"k2";
        let k3 = b"k3";
        let k4 = b"k4";
        let k5 = b"k5";

        let v1 = b"v1";
        let v2 = b"v2";
        let v3 = b"v3";

        // MVCC Versions
        // commit_ts  writes
        // 20         k1: put(v1)  k2: put(v1) k3: put(v1)    k4: put(v1)
        // 40         k1: put(v2)  k2: put(v2) k3: delete     k4: delete
        // 50         k1: rollback             k3: rollback
        // 60                      k2: lock                   k4: lock
        // values:    k1: v2       k2: v2      k3: -          k4: -          k5: -

        // version 20
        for k in [k1, k2, k3, k4] {
            must_prewrite_put(&mut engine, k, v1, k1, 10);
        }
        for k in [k1, k2, k3, k4] {
            must_commit(&mut engine, k, 10, 20);
        }
        // version 40
        for k in [k1, k2] {
            must_prewrite_put(&mut engine, k, v2, k1, 30);
        }
        for k in [k3, k4] {
            must_prewrite_delete(&mut engine, k, k1, 30);
        }
        for k in [k1, k2, k3, k4] {
            must_commit(&mut engine, k, 30, 40);
        }
        // version 50
        for k in [k1, k3] {
            must_prewrite_put(&mut engine, k, v3, k1, 50);
        }
        for k in [k1, k3] {
            must_rollback(&mut engine, k, 50, false)
        }
        // version 60
        for k in [k2, k4] {
            must_prewrite_lock(&mut engine, k, v2, 51);
        }
        for k in [k2, k4] {
            must_commit(&mut engine, k, 51, 60);
        }

        for start_ts in [15, 25, 35, 45, 55, 65] {
            let for_update_ts = start_ts + 50;
            let pk = k1.to_vec();
            let keys = vec![k1, k2, k3, k4, k5];
            let expects: Vec<PessimisticLockKeyResult> = [Some(v2), Some(v2), None, None, None]
                .iter()
                .map(|v| PessimisticLockKeyResult::Value(v.map(|b| b.to_vec())))
                .collect();
            let res = pessimistic_lock(
                &mut engine,
                &mut statistics,
                keys.clone()
                    .into_iter()
                    .map(|k| (k.as_ref(), false))
                    .collect(),
                pk,
                start_ts,
                for_update_ts,
                true,
            );
            assert_eq!(res.0, expects);
            for key in keys.clone() {
                must_pessimistic_locked(&mut engine, key, start_ts, for_update_ts);
            }
            for key in keys.clone() {
                must_rollback(&mut engine, key, start_ts, false)
            }

            // single key lock test
            let start_ts = start_ts + 1;
            let for_update_ts = start_ts + 50;
            for (i, key) in keys.clone().into_iter().enumerate() {
                let pk = key.to_vec();
                let keys = vec![(key.as_ref(), false)];
                let expect = &expects[i];
                let res = pessimistic_lock(
                    &mut engine,
                    &mut statistics,
                    keys,
                    pk,
                    start_ts,
                    for_update_ts,
                    true,
                );
                assert_eq!(res.0.len(), 1);
                assert_eq!(&res.0[0], expect);
                must_pessimistic_locked(&mut engine, key, start_ts, for_update_ts);
            }
            for key in keys {
                must_rollback(&mut engine, key, start_ts, false);
            }
        }
    }

    #[test]
    fn test_shared_locks() {
        let mut engine = TestEngineBuilder::new().build().unwrap();
        let mut statistics = Statistics::default();
        let pk = b"shared-lock-pk";
        let key = b"shared-lock";
        let ctx = kvrpcpb::Context::default();

        // Acquire slock and write to engine.
        let res = must_process_acquire_pessimistic_cmd(
            &mut engine,
            &mut statistics,
            ctx.clone(),
            5,
            5,
            pk,
            key,
            true,
        );
        engine.write(&ctx, res.to_be_write).unwrap();

        // Acquire another slock on the same key, which should succeed since the slock
        // is not shrink-only.
        let res = must_process_acquire_pessimistic_cmd(
            &mut engine,
            &mut statistics,
            ctx.clone(),
            7,
            7,
            pk,
            key,
            true,
        );
        engine.write(&ctx, res.to_be_write).unwrap();

        // Acquire xlock on the same key, which should persist updated slock and return
        // KeyIsLocked.
        let res = must_process_acquire_pessimistic_cmd(
            &mut engine,
            &mut statistics,
            ctx.clone(),
            10,
            10,
            pk,
            key,
            false,
        );
        assert!(res.lock_info.is_empty());
        assert!(!res.to_be_write.modifies.is_empty());
        let lock_info = res
            .pr
            .get_key_lock_info()
            .expect("expected shared lock info");
        assert!(lock_info.is_shared_lock());

        // Write updated slock back to engine.
        engine.write(&ctx, res.to_be_write).unwrap();
        let shared_locks = must_load_shared_lock(&mut engine, key);
        assert!(shared_locks.is_shrink_only());

        // Acquire xlock again on the same key, which should trigger lock waiting.
        let res = must_process_acquire_pessimistic_cmd(
            &mut engine,
            &mut statistics,
            ctx.clone(),
            12,
            12,
            pk,
            key,
            false,
        );
        assert!(!res.lock_info.is_empty());
        assert!(res.to_be_write.modifies.is_empty());
        match &res.pr {
            ProcessResult::PessimisticLockRes { res } => {
                let res = res.as_ref().unwrap();
                assert_eq!(res.0.len(), 1);
                assert_eq!(res.0[0], PessimisticLockKeyResult::Waiting);
            }
            _ => panic!("unexpected process result"),
        }

        // Acquire slock on the same key, which should also trigger lock waiting.
        let res = must_process_acquire_pessimistic_cmd(
            &mut engine,
            &mut statistics,
            ctx.clone(),
            15,
            15,
            pk,
            key,
            true,
        );
        assert!(!res.lock_info.is_empty());
        assert!(res.to_be_write.modifies.is_empty());
        match &res.pr {
            ProcessResult::PessimisticLockRes { res } => {
                let res = res.as_ref().unwrap();
                assert_eq!(res.0.len(), 1);
                assert_eq!(res.0[0], PessimisticLockKeyResult::Waiting);
            }
            _ => panic!("unexpected process result"),
        }
    }

    #[test]
    fn test_skip_locked() {
        let mut engine = TestEngineBuilder::new().build().unwrap();
        let mut statistics = Statistics::default();

        let (k1, k2, k3) = (b"k1", b"k2", b"k3");
        must_prewrite_put(&mut engine, k1, b"v1", k1, 1);
        must_commit(&mut engine, k1, 1, 2);

        // txn1 locks k2.
        let res = acquire_skip_locked(
            &mut engine,
            &mut statistics,
            vec![k2],
            k2,
            10,
            10,
            false,
            false,
        )
        .unwrap();
        res.0[0].assert_empty();
        must_pessimistic_locked(&mut engine, k2, 10, 10);

        // txn2 locks [k1, k2, k3] with skip-locked: k2 is skipped while k1 and k3 are
        // still locked, unlike the wait/nowait behaviors which abandon the whole
        // request.
        let res = acquire_skip_locked(
            &mut engine,
            &mut statistics,
            vec![k1, k2, k3],
            k1,
            20,
            20,
            false,
            false,
        )
        .unwrap();
        assert_eq!(res.0.len(), 3);
        res.0[0].assert_empty();
        res.0[1].assert_skipped();
        res.0[2].assert_empty();
        must_pessimistic_locked(&mut engine, k1, 20, 20);
        must_pessimistic_locked(&mut engine, k3, 20, 20);
        // k2 is still locked by txn1.
        must_pessimistic_locked(&mut engine, k2, 10, 10);

        // Keys locked by the same transaction are not skipped.
        let res = acquire_skip_locked(
            &mut engine,
            &mut statistics,
            vec![k2],
            k2,
            10,
            10,
            false,
            false,
        )
        .unwrap();
        res.0[0].assert_empty();

        // With return_values, no value is returned for a skipped key.
        let res = acquire_skip_locked(
            &mut engine,
            &mut statistics,
            vec![k1, k2],
            k1,
            20,
            20,
            true,
            false,
        )
        .unwrap();
        res.0[0].assert_value(Some(b"v1"));
        res.0[1].assert_skipped();

        // Optimistic (prewrite) locks of other transactions are skipped as well.
        must_prewrite_put(&mut engine, b"k4", b"v4", b"k4", 15);
        let res = acquire_skip_locked(
            &mut engine,
            &mut statistics,
            vec![b"k4"],
            b"k4",
            20,
            20,
            false,
            false,
        )
        .unwrap();
        res.0[0].assert_skipped();

        // A write conflict on an unlocked key is not a lock and still fails the
        // request: the client is expected to retry at a newer for_update_ts.
        must_prewrite_put(&mut engine, b"k5", b"v5", b"k5", 25);
        must_commit(&mut engine, b"k5", 25, 30);
        let err = acquire_skip_locked(
            &mut engine,
            &mut statistics,
            vec![b"k5"],
            b"k5",
            20,
            20,
            false,
            false,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            Error(box ErrorInner::Mvcc(MvccError(box MvccErrorInner::WriteConflict { .. })))
        ));

        // skip_locked together with allow_lock_with_conflict is rejected.
        let err = acquire_skip_locked(
            &mut engine,
            &mut statistics,
            vec![b"k6"],
            b"k6",
            20,
            20,
            false,
            true,
        )
        .unwrap_err();
        assert!(matches!(err, Error(box ErrorInner::Other(_))));

        // Per-key results are reported in request order even when the keys are not
        // sorted: txn at ts 30 locks k8, and a skip-locked request for [k9, k7, k8]
        // must report [Normal, Normal, Skipped].
        let res = acquire_skip_locked(
            &mut engine,
            &mut statistics,
            vec![b"k8"],
            b"k8",
            30,
            30,
            false,
            false,
        )
        .unwrap();
        res.0[0].assert_empty();
        let res = acquire_skip_locked(
            &mut engine,
            &mut statistics,
            vec![b"k9", b"k7", b"k8"],
            b"k9",
            40,
            40,
            false,
            false,
        )
        .unwrap();
        assert_eq!(res.0.len(), 3);
        res.0[0].assert_empty();
        res.0[1].assert_empty();
        res.0[2].assert_skipped();
        must_pessimistic_locked(&mut engine, b"k9", 40, 40);
        must_pessimistic_locked(&mut engine, b"k7", 40, 40);
        must_pessimistic_locked(&mut engine, b"k8", 30, 30);
    }

    #[test]
    fn test_skip_locked_shared_locks() {
        let mut engine = TestEngineBuilder::new().build().unwrap();
        let mut statistics = Statistics::default();
        let pk = b"pk";
        let key = b"shared";
        let ctx = kvrpcpb::Context::default();

        // txn1 acquires a shared lock on the key.
        let res = must_process_acquire_pessimistic_cmd(
            &mut engine,
            &mut statistics,
            ctx.clone(),
            5,
            5,
            pk,
            key,
            true,
        );
        engine.write(&ctx, res.to_be_write).unwrap();

        // An exclusive skip-locked request skips the key without converting the shared
        // locks to shrink-only, since a skip-locked reader must not mutate other
        // transactions' lock state.
        let res = acquire_skip_locked(
            &mut engine,
            &mut statistics,
            vec![key],
            pk,
            10,
            10,
            false,
            false,
        )
        .unwrap();
        res.0[0].assert_skipped();
        let shared_locks = must_load_shared_lock(&mut engine, key);
        assert!(!shared_locks.is_shrink_only());
    }

    /// Processes an `AcquirePessimisticLock` command with `skip_locked` set,
    /// asserting that no key enters the lock waiting queue, and applies its
    /// writes to the engine.
    fn acquire_skip_locked<E: Engine>(
        engine: &mut E,
        statistics: &mut Statistics,
        keys: Vec<&[u8]>,
        pk: &[u8],
        start_ts: u64,
        for_update_ts: u64,
        return_values: bool,
        allow_lock_with_conflict: bool,
    ) -> Result<PessimisticLockResults> {
        let ctx = kvrpcpb::Context::default();
        let snap = engine.snapshot(Default::default()).unwrap();
        let concurrency_manager = ConcurrencyManager::new_for_test(start_ts.into());
        let cmd = AcquirePessimisticLock::new(
            keys.into_iter()
                .map(|k| (Key::from_raw(k), false, false))
                .collect(),
            pk.to_vec(),
            start_ts.into(),
            3000,
            false,
            for_update_ts.into(),
            Some(WaitTimeout::Default),
            return_values,
            TimeStamp::zero(),
            false,
            false,
            allow_lock_with_conflict,
            true,
            ctx.clone(),
        );
        let context = WriteContext {
            lock_mgr: &MockLockManager::new(),
            concurrency_manager,
            extra_op: ExtraOp::Noop,
            statistics,
            async_apply_prewrite: false,
            raw_ext: None,
            txn_status_cache: Arc::new(TxnStatusCache::new_for_test()),
        };
        let res = cmd.cmd.process_write(snap, context)?;
        // Skipped keys never enter the lock waiting queue, even though a wait timeout
        // is set on the request.
        assert!(res.lock_info.is_empty());
        if !res.to_be_write.modifies.is_empty() {
            engine.write(&ctx, res.to_be_write).unwrap();
        }
        match res.pr {
            ProcessResult::PessimisticLockRes { res } => Ok(res.unwrap()),
            _ => unreachable!(),
        }
    }

    fn must_process_acquire_pessimistic_cmd<E: Engine>(
        engine: &mut E,
        statistics: &mut Statistics,
        ctx: kvrpcpb::Context,
        start_ts: u64,
        for_update_ts: u64,
        pk: &[u8],
        key: &[u8],
        shared: bool,
    ) -> WriteResult {
        let snap = engine.snapshot(Default::default()).unwrap();
        let concurrency_manager = ConcurrencyManager::new_for_test(start_ts.into());
        let cmd = AcquirePessimisticLock::new(
            vec![(Key::from_raw(key), false, shared)],
            pk.to_vec(),
            start_ts.into(),
            3000,
            false,
            for_update_ts.into(),
            Some(WaitTimeout::Default),
            false,
            TimeStamp::zero(),
            false,
            false,
            false,
            false,
            ctx,
        );
        let context = WriteContext {
            lock_mgr: &MockLockManager::new(),
            concurrency_manager,
            extra_op: ExtraOp::Noop,
            statistics,
            async_apply_prewrite: false,
            raw_ext: None,
            txn_status_cache: Arc::new(TxnStatusCache::new_for_test()),
        };
        cmd.cmd.process_write(snap, context).unwrap()
    }
}
