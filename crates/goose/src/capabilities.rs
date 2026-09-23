//! Capabilities the model may be allowed to use in a session, and the grants
//! that allow them.
//!
//! A capability is off until the user turns it on with `/permit <capability>`
//! and off again with `/deny <capability>`. Grants live in the session's own
//! extension data under [`PERMISSIONS_EXTENSION`], a key the agent owns: no
//! tool and no extension can write it, so a model cannot grant itself a
//! capability, and a session can never reach into another session's grants.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

use crate::session::extension_data::{ExtensionData, ExtensionState};

/// The extension-data key grants are stored under. Owned by the agent.
pub const PERMISSIONS_EXTENSION: &str = "permissions";

/// Let the model compact the session's history on its own judgement.
pub const SESSION_MODIFICATION: &str = "session-modification";

pub struct CapabilityDef {
    pub name: &'static str,
    pub description: &'static str,
}

pub static CAPABILITIES: &[CapabilityDef] = &[CapabilityDef {
    name: SESSION_MODIFICATION,
    description: "Let the model compact the conversation when it judges the work needs it",
}];

pub fn list_capabilities() -> &'static [CapabilityDef] {
    CAPABILITIES
}

pub fn find_capability(name: &str) -> Option<&'static CapabilityDef> {
    CAPABILITIES.iter().find(|def| def.name == name)
}

pub fn capability_names() -> Vec<&'static str> {
    CAPABILITIES.iter().map(|def| def.name).collect()
}

/// The capabilities a session has been granted. Nothing is granted by default.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionPermissions {
    #[serde(default)]
    granted: BTreeSet<String>,
}

impl SessionPermissions {
    pub fn is_granted(&self, capability: &str) -> bool {
        self.granted.contains(capability)
    }

    pub fn grant(&mut self, capability: &str) -> bool {
        self.granted.insert(capability.to_string())
    }

    pub fn revoke(&mut self, capability: &str) -> bool {
        self.granted.remove(capability)
    }

    pub fn granted(&self) -> Vec<&str> {
        self.granted.iter().map(String::as_str).collect()
    }

    /// Read the grants out of a session's extension data. An absent or
    /// unreadable record means nothing is granted.
    pub fn read(extension_data: &ExtensionData) -> Self {
        Self::from_extension_data(extension_data).unwrap_or_default()
    }

    /// Write the grants back, leaving other extensions' data alone.
    pub fn write_into(&self, extension_data: &mut ExtensionData) -> Result<()> {
        self.to_extension_data(extension_data)
    }
}

impl ExtensionState for SessionPermissions {
    const EXTENSION_NAME: &'static str = PERMISSIONS_EXTENSION;
    const VERSION: &'static str = "v0";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nothing_is_granted_by_default() {
        let permissions = SessionPermissions::read(&ExtensionData::new());

        assert!(!permissions.is_granted(SESSION_MODIFICATION));
        assert!(permissions.granted().is_empty());
    }

    #[test]
    fn granting_and_revoking_round_trips_through_extension_data() {
        let mut extension_data = ExtensionData::new();
        let mut permissions = SessionPermissions::default();

        assert!(permissions.grant(SESSION_MODIFICATION));
        assert!(!permissions.grant(SESSION_MODIFICATION));
        permissions.write_into(&mut extension_data).unwrap();

        let reloaded = SessionPermissions::read(&extension_data);
        assert!(reloaded.is_granted(SESSION_MODIFICATION));
        assert_eq!(reloaded.granted(), vec![SESSION_MODIFICATION]);

        permissions.revoke(SESSION_MODIFICATION);
        permissions.write_into(&mut extension_data).unwrap();
        assert!(!SessionPermissions::read(&extension_data).is_granted(SESSION_MODIFICATION));
    }

    #[test]
    fn unreadable_records_are_treated_as_no_grants() {
        let mut extension_data = ExtensionData::new();
        extension_data.set_extension_state(
            PERMISSIONS_EXTENSION,
            "v0",
            serde_json::json!("not a permission record"),
        );

        assert!(!SessionPermissions::read(&extension_data).is_granted(SESSION_MODIFICATION));
    }

    #[test]
    fn listing_reports_every_capability_and_its_description() {
        let listed = list_capabilities();

        assert!(listed.iter().any(|def| def.name == SESSION_MODIFICATION));
        assert!(listed.iter().all(|def| !def.description.is_empty()));
        assert!(find_capability("not-a-capability").is_none());
    }
}
