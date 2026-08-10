import torch

from tension_grade.model import ModelConfig, TensionGradeTransformer, grade_loss


def test_model_output_and_loss() -> None:
    config = ModelConfig(num_placements=20, num_grades=8, width=32, heads=4, layers=2)
    model = TensionGradeTransformer(config)
    logits = model(
        placement_ids=torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
        roles=torch.tensor([[0, 1, 3, 0], [0, 3, 0, 0]]),
        coordinates=torch.rand(2, 4, 2),
        mask=torch.tensor([[True, True, True, False], [True, True, False, False]]),
        angles=torch.tensor([0, 4]),
    )
    assert logits.shape == (2, 8)
    loss = grade_loss(logits, torch.tensor([2, 6]), torch.ones(2))
    assert torch.isfinite(loss)
    loss.backward()
