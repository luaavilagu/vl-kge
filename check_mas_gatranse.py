"""Check MASGATransE construction, scoring, and fusion with tiny CPU tensors.

Run from the repository root with: python check_mas_gatranse.py
This check scores one synthetic triple without training or writing a checkpoint.
"""

import torch

from vlkge.models.mas_gatranse import MASGATransE


def main():
    torch.manual_seed(42)

    # Supplied feature matrices have shape (3, 2); structure is learned separately.
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
        use_structural=True,
        use_visual=True,
        use_textual=True,
        freeze_visual=True,
        freeze_textual=True,
        visual_proj=False,
        textual_proj=False,
        weights=(1/3, 1/3, 1/3),
        device=torch.device("cpu"),
    )

    # Relative contributions are a vector buffer, not gradient-trained parameters.
    torch.testing.assert_close(model.modality_weights, torch.full((3,), 1/3))
    assert model.modality_weights.shape == torch.Size([3]), "Expected three weights."
    assert model.modality_weights.dtype == torch.float32, "Weights must be float32."
    assert model.modality_weights.device.type == "cpu", "Expected CPU weights."
    assert not model.modality_weights.requires_grad, "Weights must not require gradients."
    assert "modality_weights" in dict(model.named_buffers()), "Weights are not a buffer."
    assert "modality_weights" not in dict(model.named_parameters()), (
        "Fusion weights must not be optimizer parameters."
    )
    assert "modality_weights" in model.state_dict(), "Weights are missing from model state."

    # Supplied entity features stay frozen; relation embeddings remain trainable.
    assert not model.visual_embeddings.weight.requires_grad
    assert not model.textual_embeddings.weight.requires_grad
    assert model.relation_embeddings.weight.requires_grad
    assert model.entity_embeddings.weight.requires_grad

    print("PASS: Three-modality MASGATransE construction and initial weights.")
    print(f"Visual feature shape: {tuple(visual_features.shape)}")
    print(f"Textual feature shape: {tuple(textual_features.shape)}")
    print(f"Weights (visual, textual, structural): {model.modality_weights.tolist()}")
    print("Visual/textual features are frozen; structural entities and relations are trainable.")

    # One synthetic triple: entity 0, relation 0, entity 1.
    head = torch.tensor([0], dtype=torch.long)
    relation = torch.tensor([0], dtype=torch.long)
    tail = torch.tensor([1], dtype=torch.long)
    original_weights_buffer = model.modality_weights

    model.eval()
    try:
        with torch.no_grad():
            visual_score = model.score_visual_evidence(head, relation, tail)
            textual_score = model.score_textual_evidence(head, relation, tail)
            structural_score = model.score_structural_evidence(head, relation, tail)

            for name, score in (
                ("visual", visual_score), ("textual", textual_score), ("structural", structural_score)
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
            print(f"Structural score: {structural_score.item():.6f}")

            # Each vertex selects one modality; equal thirds averages all three.
            cases = (
                ((1/3, 1/3, 1/3), (visual_score + textual_score + structural_score) / 3, "equal thirds"),
                ((1., 0., 0.), visual_score, "visual only"),
                ((0., 1., 0.), textual_score, "textual only"),
                ((0., 0., 1.), structural_score, "structural only"),
            )
            for weights, expected, label in cases:
                model.set_weights(weights)
                torch.testing.assert_close(model.modality_weights, torch.tensor(weights))
                actual = model(head, relation, tail)
                assert actual.shape == torch.Size([1]), "Expected one fused score."
                assert torch.isfinite(actual).all().item(), "Fused score must be finite."
                torch.testing.assert_close(actual, expected)
                assert model.modality_weights is original_weights_buffer, (
                    "set_weights must update the existing buffer in place."
                )
                print(f"PASS: {label} (weights={weights}, score={actual.item():.6f}).")
    finally:
        model.set_weights((1/3, 1/3, 1/3))

    torch.testing.assert_close(model.modality_weights, torch.full((3,), 1/3))
    assert dict(model.named_buffers())["modality_weights"] is original_weights_buffer
    assert not model.modality_weights.requires_grad
    print("PASS: Equal thirds restored; the original registered buffer is preserved.")


if __name__ == "__main__":
    main()
