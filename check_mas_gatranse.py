"""Check MASGATransE construction, scoring, and fusion with tiny CPU tensors.

Run from the repository root with: python check_mas_gatranse.py
This check scores one synthetic triple without training or writing a checkpoint.
"""

import torch

from vlkge.models.mas_gatranse import MASGATransE


def main():
    torch.manual_seed(42)

    # Each row is one entity's embedding; both modalities have shape (3, 2).
    visual_features = torch.tensor(
        [[1, 0], [0, 1], [-1, 0]], dtype=torch.float32
    )
    textual_features = torch.tensor(
        [[0, 1], [1, 0], [0, -1]], dtype=torch.float32
    )
    entity_to_index = {0: 0, 1: 1, 2: 2}

    model = MASGATransE(
        num_entities=3,
        num_relations=1,
        embedding_dim=2,
        visual_features=visual_features,
        textual_features=textual_features,
        relation_features=None,
        visual_entity_to_index=entity_to_index.copy(),
        textual_entity_to_index=entity_to_index.copy(),
        fusion_mode="average",
        use_structural=False,
        use_visual=True,
        use_textual=True,
        freeze_visual=True,
        freeze_textual=True,
        visual_proj=False,
        textual_proj=False,
        alpha=0.5,
        device=torch.device("cpu"),
    )

    # Check the initial alpha and its registration without modifying it.
    torch.testing.assert_close(model.alpha_tensor, torch.tensor(0.5))
    assert model.alpha_tensor.shape == torch.Size([]), "Alpha must be scalar."
    assert model.alpha_tensor.dtype == torch.float32, "Alpha must be float32."
    assert model.alpha_tensor.device.type == "cpu", "Expected CPU alpha."
    assert not model.alpha_tensor.requires_grad, "Alpha must not require gradients."
    assert "alpha_tensor" in dict(model.named_buffers()), "Alpha is not a buffer."
    assert "alpha_tensor" not in dict(model.named_parameters()), (
        "Alpha must not be an optimizer parameter."
    )
    assert "alpha_tensor" in model.state_dict(), "Alpha is missing from model state."

    # Supplied entity features stay frozen; relation embeddings remain trainable.
    assert not model.visual_embeddings.weight.requires_grad
    assert not model.textual_embeddings.weight.requires_grad
    assert model.relation_embeddings.weight.requires_grad

    print("PASS: MASGATransE construction and initial alpha checks.")
    print(f"Visual feature shape: {tuple(visual_features.shape)}")
    print(f"Textual feature shape: {tuple(textual_features.shape)}")
    print(f"Alpha: {model.alpha_tensor.item()}")
    print(f"Alpha shape: {tuple(model.alpha_tensor.shape)} (scalar)")
    print(f"Alpha requires gradients: {model.alpha_tensor.requires_grad}")
    print("Entity features are frozen; relation embeddings are trainable.")

    # One synthetic triple: entity 0, relation 0, entity 1.
    head = torch.tensor([0], dtype=torch.long)
    relation = torch.tensor([0], dtype=torch.long)
    tail = torch.tensor([1], dtype=torch.long)
    original_alpha_buffer = model.alpha_tensor

    model.eval()
    try:
        with torch.no_grad():
            visual_score = model.score_visual_evidence(head, relation, tail)
            textual_score = model.score_textual_evidence(head, relation, tail)

            for name, score in (
                ("visual", visual_score), ("textual", textual_score)
            ):
                assert score.shape == torch.Size([1]), (
                    f"Expected one {name} score, got shape {tuple(score.shape)}."
                )
                assert torch.isfinite(score).all().item(), (
                    f"The {name} score must be finite."
                )

            print("PASS: Separate modality scores have shape (1,) and are finite.")
            print(f"Visual score: {visual_score.item():.6f}")
            print(f"Textual score: {textual_score.item():.6f}")

            # Each endpoint must select exactly one modality; 0.5 averages both.
            cases = (
                (0.5, (visual_score + textual_score) / 2, "equal-weight fusion"),
                (1.0, visual_score, "visual-only fusion"),
                (0.0, textual_score, "textual-only fusion"),
            )
            for alpha, expected, label in cases:
                model.set_alpha(alpha)
                torch.testing.assert_close(model.alpha_tensor, torch.tensor(alpha))
                actual = model(head, relation, tail)
                assert actual.shape == torch.Size([1]), "Expected one fused score."
                assert torch.isfinite(actual).all().item(), "Fused score must be finite."
                torch.testing.assert_close(actual, expected)
                assert model.alpha_tensor is original_alpha_buffer, (
                    "set_alpha must update the existing buffer in place."
                )
                print(f"PASS: {label} (alpha={alpha}, score={actual.item():.6f}).")
    finally:
        model.set_alpha(0.5)

    torch.testing.assert_close(model.alpha_tensor, torch.tensor(0.5))
    assert dict(model.named_buffers())["alpha_tensor"] is original_alpha_buffer
    assert not model.alpha_tensor.requires_grad
    print("PASS: Alpha restored to 0.5 and remains the original registered buffer.")


if __name__ == "__main__":
    main()
