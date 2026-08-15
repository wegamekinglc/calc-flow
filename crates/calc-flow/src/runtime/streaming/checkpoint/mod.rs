pub(crate) mod coordinator;
mod storage;

pub(crate) use storage::{ManagedCheckpointRuntime, OpenedManagedCheckpointRuntime};

#[cfg(test)]
mod tests {
    use super::ManagedCheckpointRuntime;

    #[test]
    fn managed_checkpoint_runtime_construction_is_pure_and_redacts_its_root() {
        let directory = tempfile::tempdir().unwrap();
        let sentinel = directory.path().join("credential-secret-managed-root");

        let runtime = ManagedCheckpointRuntime::new(&sentinel).unwrap();

        assert!(!sentinel.exists());
        for rendered in [format!("{runtime:?}"), format!("{runtime:#?}")] {
            assert!(!rendered.contains("credential-secret-managed-root"));
            assert!(!rendered.contains(&sentinel.display().to_string()));
        }
    }

    #[test]
    fn managed_checkpoint_runtime_rejects_an_empty_root_without_io() {
        let error = ManagedCheckpointRuntime::new(std::path::PathBuf::new()).unwrap_err();

        assert!(matches!(
            error,
            crate::CalcFlowError::InvalidArgument { ref field, ref message }
                if field == "managed_checkpoint_root" && message == "must not be empty"
        ));
    }

    #[tokio::test]
    async fn managed_checkpoint_runtime_prepares_fixed_children_and_leases_the_root() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("managed");
        let runtime = ManagedCheckpointRuntime::new(&root).unwrap();

        let opened = runtime
            .open(&crate::CancellationToken::new())
            .await
            .unwrap();

        // The runtime stores canonicalized paths; on Windows the lexical temp
        // path uses 8.3 short names while the canonical form is verbatim.
        let canonical_root = std::fs::canonicalize(&root).unwrap();
        assert_eq!(opened.state_root_for_test(), canonical_root.join("state"));
        assert_eq!(
            opened.manifest_root_for_test(),
            canonical_root.join("manifests")
        );
        assert!(opened.state_root_for_test().is_dir());
        assert!(opened.manifest_root_for_test().is_dir());

        let same_root = ManagedCheckpointRuntime::new(&root).unwrap();
        let conflict = same_root
            .open(&crate::CancellationToken::new())
            .await
            .unwrap_err();
        assert!(matches!(conflict, crate::CalcFlowError::Conflict { .. }));

        drop(opened);
        ManagedCheckpointRuntime::new(&root)
            .unwrap()
            .open(&crate::CancellationToken::new())
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn managed_checkpoint_runtime_preflight_errors_redact_sensitive_paths() {
        use std::error::Error as _;

        let directory = tempfile::tempdir().unwrap();
        let sentinel = directory.path().join("credential-secret-invalid-root");
        std::fs::write(&sentinel, b"not a directory").unwrap();
        let runtime = ManagedCheckpointRuntime::new(&sentinel).unwrap();

        let error = runtime
            .open(&crate::CancellationToken::new())
            .await
            .unwrap_err();

        for rendered in [
            error.to_string(),
            format!("{error:?}"),
            format!("{error:#?}"),
        ] {
            assert!(!rendered.contains("credential-secret-invalid-root"));
            assert!(!rendered.contains(&sentinel.display().to_string()));
        }
        assert!(error.source().is_none());
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn managed_checkpoint_runtime_resolves_symlink_aliases_before_leasing() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("managed");
        let alias = directory.path().join("managed-alias");
        let opened = ManagedCheckpointRuntime::new(&root)
            .unwrap()
            .open(&crate::CancellationToken::new())
            .await
            .unwrap();
        std::os::unix::fs::symlink(&root, &alias).unwrap();

        let conflict = ManagedCheckpointRuntime::new(&alias)
            .unwrap()
            .open(&crate::CancellationToken::new())
            .await
            .unwrap_err();

        assert!(matches!(conflict, crate::CalcFlowError::Conflict { .. }));
        drop(opened);
    }

    #[tokio::test]
    async fn managed_checkpoint_runtime_rejects_nested_managed_roots() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("managed");
        let opened = ManagedCheckpointRuntime::new(&root)
            .unwrap()
            .open(&crate::CancellationToken::new())
            .await
            .unwrap();

        let descendant = ManagedCheckpointRuntime::new(root.join("nested"))
            .unwrap()
            .open(&crate::CancellationToken::new())
            .await
            .unwrap_err();

        assert!(matches!(descendant, crate::CalcFlowError::Conflict { .. }));
        drop(opened);
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn managed_checkpoint_runtime_rejects_subprocess_descendant_without_temporary_storage() {
        const CHILD_ROOT: &str = "CALC_FLOW_CHECKPOINT_LEASE_CHILD_ROOT";

        if let Some(root) = std::env::var_os(CHILD_ROOT) {
            let error = ManagedCheckpointRuntime::new(root)
                .unwrap()
                .open(&crate::CancellationToken::new())
                .await
                .unwrap_err();
            assert!(matches!(error, crate::CalcFlowError::Conflict { .. }));
            return;
        }

        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("managed");
        let opened = ManagedCheckpointRuntime::new(&root)
            .unwrap()
            .open(&crate::CancellationToken::new())
            .await
            .unwrap();
        let invalid_temporary_directory = directory.path().join("not-a-directory");
        std::fs::write(&invalid_temporary_directory, b"sentinel").unwrap();
        let output = std::process::Command::new("/proc/self/exe")
            .args([
                "--exact",
                "runtime::streaming::checkpoint::tests::managed_checkpoint_runtime_rejects_subprocess_descendant_without_temporary_storage",
                "--nocapture",
            ])
            .env(CHILD_ROOT, root.join("nested"))
            .env("TMPDIR", invalid_temporary_directory)
            .output()
            .unwrap();

        assert!(
            output.status.success(),
            "child failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        drop(opened);
    }
}
