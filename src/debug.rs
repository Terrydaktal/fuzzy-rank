#[derive(Clone, Debug)]
pub struct DebugStage {
    pub name: &'static str,
    pub value: String,
}

impl DebugStage {
    pub fn new(name: &'static str, value: impl Into<String>) -> Self {
        Self {
            name,
            value: value.into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct DebugReport {
    pub matched: bool,
    pub summary: String,
    pub stages: Vec<DebugStage>,
}

impl DebugReport {
    pub fn matched(summary: impl Into<String>, stages: Vec<DebugStage>) -> Self {
        Self {
            matched: true,
            summary: summary.into(),
            stages,
        }
    }

    pub fn no_match(summary: impl Into<String>, stages: Vec<DebugStage>) -> Self {
        Self {
            matched: false,
            summary: summary.into(),
            stages,
        }
    }
}
