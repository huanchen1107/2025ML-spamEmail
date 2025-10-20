## MODIFIED Requirements

### Requirement: Spam Email Classification
The system MUST provide options to optimize for higher recall on spam while maintaining acceptable precision.

#### Scenario: Improved recall at acceptable precision
- WHEN training is performed with enhanced features or class weighting
- THEN Recall on held-out test >= 0.93
- AND Precision remains >= 0.90 (dataset-dependent)