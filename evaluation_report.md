# Legal Summary Evaluation Report
Total examples evaluated: 169
Overall accuracy: 0.7456

## Accuracy by Attribute
| Attribute | Accuracy |
| --- | --- |
| plaintiff | 0.8521 |
| defendant | 0.9231 |
| filing_date | 0.7751 |
| court_name | 0.9053 |
| statutory | 0.8166 |
| holding | 0.7811 |
| outcome | 0.5325 |
| judge | 0.5325 |
| outcome_date | 0.5917 |

## Label Distribution

### plaintiff
| Label | Count |
| --- | --- |
| included_complete | 150 |
| included_incomplete | 15 |
| included_contradiction | 2 |
| included_extra | 2 |

### defendant
| Label | Count |
| --- | --- |
| included_complete | 160 |
| included_extra | 4 |
| included_incomplete | 4 |
| included_contradiction | 1 |

### filing_date
| Label | Count |
| --- | --- |
| included_contradiction | 12 |
| not_included | 9 |
| included_complete | 147 |
| included_incomplete | 1 |

### court_name
| Label | Count |
| --- | --- |
| included_complete | 155 |
| included_extra | 11 |
| included_incomplete | 2 |
| not_included | 1 |

### statutory
| Label | Count |
| --- | --- |
| included_complete | 144 |
| included_contradiction | 9 |
| included_incomplete | 12 |
| not_included | 2 |
| included_extra | 2 |

### holding
| Label | Count |
| --- | --- |
| included_complete | 146 |
| not_included | 10 |
| included_contradiction | 9 |
| included_incomplete | 4 |

### outcome
| Label | Count |
| --- | --- |
| included_incomplete | 30 |
| included_complete | 82 |
| included_contradiction | 2 |
| not_included | 49 |
| included_extra | 6 |

### judge
| Label | Count |
| --- | --- |
| not_included | 8 |
| included_incomplete | 15 |
| included_extra | 60 |
| included_complete | 83 |
| included_contradiction | 3 |

### outcome_date
| Label | Count |
| --- | --- |
| included_complete | 105 |
| included_contradiction | 38 |
| not_included | 14 |
| included_extra | 7 |
| included_incomplete | 5 |

## Confusion Matrices

### plaintiff
| Human ↓ / LLM → | included_complete | included_contradiction | included_extra | included_incomplete |
| --- | --- | --- | --- | --- |
| included_complete | 144 | 6 | 0 | 0 | |
| included_contradiction | 2 | 0 | 0 | 0 | |
| included_extra | 2 | 0 | 0 | 0 | |
| included_incomplete | 15 | 0 | 0 | 0 | |

### defendant
| Human ↓ / LLM → | included_complete | included_contradiction | included_extra | included_incomplete |
| --- | --- | --- | --- | --- |
| included_complete | 156 | 4 | 0 | 0 | |
| included_contradiction | 1 | 0 | 0 | 0 | |
| included_extra | 3 | 1 | 0 | 0 | |
| included_incomplete | 4 | 0 | 0 | 0 | |

### filing_date
| Human ↓ / LLM → | included_complete | included_contradiction | included_incomplete | not_included |
| --- | --- | --- | --- | --- |
| included_complete | 116 | 31 | 0 | 0 | |
| included_contradiction | 1 | 11 | 0 | 0 | |
| included_incomplete | 1 | 0 | 0 | 0 | |
| not_included | 4 | 1 | 0 | 4 | |

### court_name
| Human ↓ / LLM → | included_complete | included_contradiction | included_extra | included_incomplete | not_included |
| --- | --- | --- | --- | --- | --- |
| included_complete | 152 | 2 | 0 | 0 | 1 | |
| included_contradiction | 0 | 0 | 0 | 0 | 0 | |
| included_extra | 9 | 1 | 0 | 0 | 1 | |
| included_incomplete | 1 | 1 | 0 | 0 | 0 | |
| not_included | 0 | 0 | 0 | 0 | 1 | |

### statutory
| Human ↓ / LLM → | included_complete | included_contradiction | included_extra | included_incomplete | not_included |
| --- | --- | --- | --- | --- | --- |
| included_complete | 137 | 7 | 0 | 0 | 0 | |
| included_contradiction | 8 | 1 | 0 | 0 | 0 | |
| included_extra | 2 | 0 | 0 | 0 | 0 | |
| included_incomplete | 12 | 0 | 0 | 0 | 0 | |
| not_included | 2 | 0 | 0 | 0 | 0 | |

### holding
| Human ↓ / LLM → | included_complete | included_contradiction | included_incomplete | not_included |
| --- | --- | --- | --- | --- |
| included_complete | 126 | 6 | 3 | 11 | |
| included_contradiction | 7 | 1 | 0 | 1 | |
| included_incomplete | 4 | 0 | 0 | 0 | |
| not_included | 5 | 0 | 0 | 5 | |

### outcome
| Human ↓ / LLM → | included_complete | included_contradiction | included_extra | included_incomplete | not_included |
| --- | --- | --- | --- | --- | --- |
| included_complete | 66 | 2 | 0 | 10 | 4 | |
| included_contradiction | 0 | 0 | 0 | 1 | 1 | |
| included_extra | 4 | 0 | 0 | 2 | 0 | |
| included_incomplete | 18 | 0 | 0 | 12 | 0 | |
| not_included | 19 | 0 | 0 | 18 | 12 | |

### judge
| Human ↓ / LLM → | included_complete | included_contradiction | included_extra | included_incomplete | not_included |
| --- | --- | --- | --- | --- | --- |
| included_complete | 81 | 2 | 0 | 0 | 0 | |
| included_contradiction | 2 | 1 | 0 | 0 | 0 | |
| included_extra | 60 | 0 | 0 | 0 | 0 | |
| included_incomplete | 14 | 1 | 0 | 0 | 0 | |
| not_included | 0 | 0 | 0 | 0 | 8 | |

### outcome_date
| Human ↓ / LLM → | included_complete | included_contradiction | included_extra | included_incomplete | not_included |
| --- | --- | --- | --- | --- | --- |
| included_complete | 82 | 23 | 0 | 0 | 0 | |
| included_contradiction | 21 | 11 | 0 | 1 | 5 | |
| included_extra | 5 | 2 | 0 | 0 | 0 | |
| included_incomplete | 4 | 0 | 0 | 0 | 1 | |
| not_included | 5 | 2 | 0 | 0 | 7 | |
