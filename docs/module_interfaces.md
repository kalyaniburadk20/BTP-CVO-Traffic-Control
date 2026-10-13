# Module Interfaces

This document defines the interfaces between the traffic-state,
observation, belief, and controller modules.

---

## 1. Traffic State Module

### File

`src/perception/traffic_state.py`

### Purpose

Represent the current traffic condition obtained from SUMO.

### Input

SUMO traffic information:

- Vehicle IDs
- Queue lengths
- Waiting times
- Current traffic-light phase
- Vehicle count

### Output

`TrafficState`

### Representation

```text
TrafficState
├── north_queue
├── south_queue
├── east_queue
├── west_queue
├── north_waiting
├── south_waiting
├── east_waiting
├── west_waiting
├── current_phase
└── vehicle_count