"""Evaluate the MASNeuralTransE visual-textual scoring policy on validation data."""

import torch
from torch.utils.data import DataLoader

from vlkge import helpers, utils
from vlkge.dataloader import KGDataset, KnowledgeGraphDataLoader
from vlkge.mas.run_mas import load_mas_model, parse_args


def main():
    args = parse_args(require_candidate=False)
    utils.set_seed(args.seed)
    device = utils.set_gpu() if args.cuda else torch.device("cpu")

    print("Loading dataset...")
    data_loader = KnowledgeGraphDataLoader(
        data_path=args.data_path,
        dataset_name=args.dataset,
        exclude_relations=args.exclude_relations,
        exclude_relations_eval=args.exclude_relations_eval,
        add_inverse_relations=args.add_inverse_relations,
        use_per_relation_candidates=args.use_per_relation_candidates,
        artist2artist_relations=args.artist2artist_relations,
        bidirectional_eval=args.bidirectional_eval,
    )
    entity_to_id, relation_to_id = data_loader.get_entities_and_relations()

    _, validation_data, _ = data_loader.split_data()
    validation_dataset = KGDataset(validation_data, entity_to_id, relation_to_id)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    print("Building evaluation mappings...")
    filter_map = data_loader.compute_filter_map()
    model = load_mas_model(args, entity_to_id, relation_to_id, device)

    relation_candidates = (
        data_loader.relation_to_valid_tails_eval_val
        if args.use_per_relation_candidates
        else None
    )
    mrr, hits, relation_metrics = helpers.evaluate_kge(
        model=model,
        dataloader=validation_loader,
        filter_map=filter_map,
        ks=args.top_k,
        bidirectional=args.bidirectional_eval,
        relation_to_valid_tails=relation_candidates,
        device=device,
    )

    id_to_relation = {relation_id: name for name, relation_id in relation_to_id.items()}

    print(f"\n{'=' * 80}")
    print("FINAL VALIDATION RESULTS:")
    print(f"  MRR: {mrr:.4f}")
    for k in args.top_k:
        print(f"  Hits@{k}: {hits[k]:.4f}")
    print(f"{'=' * 80}")

    print("\nPer-Relation Validation Metrics:")
    print("-" * 80)
    for relation_id, metrics in relation_metrics.items():
        relation_name = id_to_relation[relation_id]
        hits_text = " | ".join(
            f"Hits@{k}: {metrics[f'Hits@{k}']:.4f}" for k in args.top_k
        )
        print(f"{relation_name:30}: MRR: {metrics['MRR']:.4f} | {hits_text}")
    print("-" * 80)
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
