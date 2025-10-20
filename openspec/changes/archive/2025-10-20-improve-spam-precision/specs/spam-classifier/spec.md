## MODIFIED Requirements

### Requirement: Spam Email Classification
The system MUST achieve Precision >= 0.90 while maintaining Recall >= 0.93 under the recommended configuration.

#### Scenario: Precision restored with high recall
- WHEN training is performed with tuned parameters and threshold
- THEN Precision on held-out test >= 0.90
- AND Recall remains >= 0.93