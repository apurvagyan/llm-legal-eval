# Legal Summary Evaluation Report
Total examples evaluated: 169
Overall accuracy: 0.7173

## Accuracy by Attribute
| Attribute | Accuracy |
| --- | --- |
| plaintiff | 0.8580 |
| defendant | 0.8876 |
| filing_date | 0.7811 |
| court_name | 0.8639 |
| statutory | 0.7278 |
| holding | 0.7633 |
| outcome_date | 0.5562 |
| outcome | 0.5148 |
| judge | 0.5030 |

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
| included_complete | 145 | 4 | 0 | 1 | |
| included_contradiction | 2 | 0 | 0 | 0 | |
| included_extra | 2 | 0 | 0 | 0 | |
| included_incomplete | 15 | 0 | 0 | 0 | |

### defendant
| Human ↓ / LLM → | included_complete | included_contradiction | included_extra | included_incomplete |
| --- | --- | --- | --- | --- |
| included_complete | 150 | 10 | 0 | 0 | |
| included_contradiction | 1 | 0 | 0 | 0 | |
| included_extra | 3 | 0 | 0 | 1 | |
| included_incomplete | 4 | 0 | 0 | 0 | |

### filing_date
| Human ↓ / LLM → | included_complete | included_contradiction | included_incomplete | not_included |
| --- | --- | --- | --- | --- |
| included_complete | 120 | 27 | 0 | 0 | |
| included_contradiction | 2 | 8 | 1 | 1 | |
| included_incomplete | 1 | 0 | 0 | 0 | |
| not_included | 3 | 2 | 0 | 4 | |

### court_name
| Human ↓ / LLM → | included_complete | included_contradiction | included_extra | included_incomplete | not_included |
| --- | --- | --- | --- | --- | --- |
| included_complete | 145 | 4 | 0 | 6 | 0 | |
| included_contradiction | 0 | 0 | 0 | 0 | 0 | |
| included_extra | 9 | 1 | 0 | 1 | 0 | |
| included_incomplete | 2 | 0 | 0 | 0 | 0 | |
| not_included | 0 | 0 | 0 | 0 | 1 | |

### statutory
| Human ↓ / LLM → | included_complete | included_contradiction | included_extra | included_incomplete | not_included |
| --- | --- | --- | --- | --- | --- |
| included_complete | 122 | 15 | 0 | 1 | 6 | |
| included_contradiction | 8 | 1 | 0 | 0 | 0 | |
| included_extra | 2 | 0 | 0 | 0 | 0 | |
| included_incomplete | 12 | 0 | 0 | 0 | 0 | |
| not_included | 2 | 0 | 0 | 0 | 0 | |

### holding
| Human ↓ / LLM → | included_complete | included_contradiction | included_incomplete | not_included |
| --- | --- | --- | --- | --- |
| included_complete | 122 | 8 | 3 | 13 | |
| included_contradiction | 6 | 2 | 0 | 1 | |
| included_incomplete | 4 | 0 | 0 | 0 | |
| not_included | 5 | 0 | 0 | 5 | |

### outcome
| Human ↓ / LLM → | included_complete | included_contradiction | included_extra | included_incomplete | not_included |
| --- | --- | --- | --- | --- | --- |
| included_complete | 77 | 3 | 0 | 1 | 1 | |
| included_contradiction | 0 | 0 | 0 | 2 | 0 | |
| included_extra | 5 | 0 | 0 | 1 | 0 | |
| included_incomplete | 27 | 0 | 0 | 3 | 0 | |
| not_included | 35 | 0 | 0 | 7 | 7 | |

### judge
| Human ↓ / LLM → | included_complete | included_contradiction | included_extra | included_incomplete | not_included |
| --- | --- | --- | --- | --- | --- |
| included_complete | 80 | 3 | 0 | 0 | 0 | |
| included_contradiction | 2 | 1 | 0 | 0 | 0 | |
| included_extra | 60 | 0 | 0 | 0 | 0 | |
| included_incomplete | 14 | 1 | 0 | 0 | 0 | |
| not_included | 3 | 1 | 0 | 0 | 4 | |

### outcome_date
| Human ↓ / LLM → | included_complete | included_contradiction | included_extra | included_incomplete | not_included |
| --- | --- | --- | --- | --- | --- |
| included_complete | 77 | 26 | 0 | 2 | 0 | |
| included_contradiction | 18 | 12 | 0 | 4 | 4 | |
| included_extra | 7 | 0 | 0 | 0 | 0 | |
| included_incomplete | 4 | 0 | 0 | 0 | 1 | |
| not_included | 5 | 3 | 0 | 1 | 5 | |
