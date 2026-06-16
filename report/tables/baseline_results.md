# Baseline Results

| Setting | Method | Backbone | Training | Test episodes | 5-way 1-shot | 5-way 5-shot |
|---|---|---|---|---:|---:|---:|
| quick baseline | ProtoNet | Conv64F | 10 epochs; 100 train episodes/epoch | 100 | 24.133% | 37.413% |
| cloud formal baseline | ProtoNet | Conv64F | 100 epochs; 1000 train episodes/epoch | 600 | 36.249% | 69.584% |
| cloud boosted baseline | ProtoNet | ResNet12 | 120 epochs; 2000 train episodes/epoch | 1000 | 73.376% | 85.945% |
