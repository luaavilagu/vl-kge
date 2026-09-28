"""
Unified training script for VL-KGE experiments
Supports: WN9-IMG, WikiArt-MKG-v1, WikiArt-MKG-v2
"""

import os
import argparse
import sys
import yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from vlkge.dataloader import KnowledgeGraphDataLoader, KGDataset
from vlkge import helpers, utils


SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent

# Backward-compatible BooleanOptionalAction for python < 3.9
try:
    from argparse import BooleanOptionalAction
except ImportError:
    class BooleanOptionalAction(argparse.Action):
        def __init__(self, option_strings, dest, default=None, required=False, help=None):
            _option_strings = []
            for option_string in option_strings:
                _option_strings.append(option_string)
                if option_string.startswith('--'):
                    _option_strings.append('--no-' + option_string[2:])
            super().__init__(
                option_strings=_option_strings,
                dest=dest,
                nargs=0,
                default=default,
                required=required,
                help=help
            )
        
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, not option_string.startswith('--no-'))

def parse_args():
    parser = argparse.ArgumentParser(
        description='VL-KGE Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Built-in dataset with visual features only
  python3 train.py --dataset wn9_img --no-textual
  
  # Custom dataset
  python3 train.py --dataset custom \\
      --data_path /path/to/data.csv \\
      --visual_features_path /path/to/visual.pkl
  
  # Using config file
  python3 train.py --config configs/wn9_img/transe_clip.yaml
  
  # CLI overrides YAML
  python3 train.py --config configs/base.yaml --no-freeze_visual
        """
    )
    
    # Config file support
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML config file (overrides other args)')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='wn9_img',
                       help='Dataset name (default: wn9_img)')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to the dataset CSV (auto-detected for built-in datasets)')
    
    # Model
    parser.add_argument('--model', type=str, default='TransE',
                       choices=['TransE', 'DistMult', 'ComplEx', 'RotatE', 'NeuralTransE'],
                       help='KGE model (default: TransE)')
    parser.add_argument('--fusion_mode', type=str, default='average',
                       choices=['average', 'concat', 'weighted', 'addition'],
                       help='Modality fusion strategy (default: average)')
    
    # Modalities    
    parser.add_argument('--use_structural', action=BooleanOptionalAction, default=True,
                       help='Use structural embeddings (default: True)')
    parser.add_argument('--use_visual', action=BooleanOptionalAction, default=True,
                       help='Use visual features (default: True)')
    parser.add_argument('--use_textual', action=BooleanOptionalAction, default=True,
                       help='Use textual features (default: True)')
    parser.add_argument('--use_relation_features', action=BooleanOptionalAction, default=False,
                       help='Use pretrained relation features (default: False)')
    
    # Advanced feature options
    parser.add_argument('--visual_proj', action=BooleanOptionalAction, default=False,
                       help='Add residual adapter for visual features (default: False)')
    parser.add_argument('--textual_proj', action=BooleanOptionalAction, default=False,
                       help='Add residual adapter for textual features (default: False)')
    parser.add_argument('--vis_adapter_dim', type=int, default=64,
                       help='Visual adapter bottleneck dimension (default: 64)')
    parser.add_argument('--txt_adapter_dim', type=int, default=64,
                       help='Textual adapter bottleneck dimension (default: 64)')
    parser.add_argument('--shared_projection', type=str, nargs='*', default=None,
                       help='Modalities to share projection layer (e.g., visual textual)')
    parser.add_argument('--normalize_before_fusion', action=BooleanOptionalAction, default=False,
                       help='L2 normalize embeddings before fusion (default: False)')
    
    # Model-specific arguments
    parser.add_argument('--freeze_visual', action=BooleanOptionalAction, default=True,
                       help='Freeze visual feature weights (default: True)')
    parser.add_argument('--freeze_textual', action=BooleanOptionalAction, default=True,
                       help='Freeze textual feature weights (default: True)')
    
    # TransE-specific
    parser.add_argument('--p_norm', type=int, default=1, choices=[1, 2],
                       help='TransE: Distance norm (1=L1, 2=L2) (default: 1)')
    parser.add_argument('--normalize_relations', action=BooleanOptionalAction, default=True,
                       help='TransE: L2 normalize relation embeddings (default: True)')
    
    # TransE and RotatE margin
    parser.add_argument('--margin', type=float, default=12.0,
                       help='TransE/RotatE: Margin for scoring (default: 12.0)')
    
    # Features paths
    parser.add_argument('--visual_features_path', type=str, default='vlkge/data/wn9_img/features/wn9_img_vf_clip.pkl',
                       help='Path to visual features (.pkl)')
    parser.add_argument('--textual_features_path', type=str, default='vlkge/data/wn9_img/features/wn9_img_tf_clip.pkl',
                       help='Path to textual features (.pkl)')
    parser.add_argument('--relation_features_path', type=str, default='vlkge/data/wn9_img/features/wn9_img_rf_clip.pkl',
                       help='Path to relation features (.pkl)')
    
    # Feature preprocessing
    parser.add_argument('--normalize_visual', action=BooleanOptionalAction, default=False,
                       help='L2 normalize visual features at load time (default: False)')
    parser.add_argument('--normalize_textual', action=BooleanOptionalAction, default=False,
                       help='L2 normalize textual features at load time (default: False)')
    parser.add_argument('--normalize_relation', action=BooleanOptionalAction, default=False,
                       help='L2 normalize relation features at load time (default: False)')
    
    # Training
    parser.add_argument('--embedding_dim', type=int, default=768,
                       help='Embedding dimension (default: 768)')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs (default: 200)')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Training batch size (default: 512)')
    parser.add_argument('--lr', type=float, default=0.1,
                       help='Learning rate (default: 0.1)')
    parser.add_argument('--num_neg_samples', type=int, default=100,
                       help='Number of negative samples per positive (default: 100)')
    parser.add_argument('--use_bernoulli', action=BooleanOptionalAction, default=False,
                       help='Use Bernoulli negative sampling (default: False)')
    parser.add_argument('--use_scheduler', action=BooleanOptionalAction, default=False,
                       help='Use learning rate scheduler (default: False)')
    parser.add_argument('--patience', type=int, default=None,
                       help='Early stopping patience in epochs (default: None)')
    parser.add_argument('--evaluate_every', type=int, default=1,
                       help='Evaluate every N epochs (default: 1)')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    
    # Evaluation
    parser.add_argument('--top_k', type=int, nargs='+', default=[1, 3, 10],
                       help='Hits@K values for evaluation (default: 1 3 10)')
    
    # System
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--no_cuda', action='store_true',
                       help='Disable CUDA (use CPU only)')
    parser.add_argument('--save_path', type=str, default=None,
                       help='Path to save best model checkpoint')
    
    # Dataset-specific options
    parser.add_argument('--use_per_relation_candidates', action=BooleanOptionalAction, default=False,
                       help='Use per-relation candidate pools for evaluation (default: False)')
    parser.add_argument('--bidirectional_eval', action=BooleanOptionalAction, default=True,
                       help='Evaluate both head and tail prediction (default: True)')
    parser.add_argument('--inductive', action=BooleanOptionalAction, default=False,
                       help='Enable inductive learning with entity masking (default: False)')
    parser.add_argument('--modality_asymmetry', action=BooleanOptionalAction, default=False,
                       help='Handle per-entity modality combinations (default: False)')
    
    # Advanced dataset options
    parser.add_argument('--exclude_relations', type=str, nargs='*', default=None,
                       help='Relations to exclude from all splits')
    parser.add_argument('--exclude_relations_eval', type=str, nargs='*', default=None,
                       help='Relations to exclude from validation/test only')
    parser.add_argument('--add_inverse_relations', type=str, nargs='*', default=None,
                       help='Add inverse relations (format: "rel1:inv1 rel2:inv2")')
    parser.add_argument('--artist2artist_relations', type=str, nargs='*', default=None,
                       help='Artist-to-artist relations for WikiArt-style datasets')
    parser.add_argument('--downsample_relation', type=str, default=None,
                       help='Relation to downsample during training')
    parser.add_argument('--downsample_fraction', type=float, default=1.0,
                       help='Fraction of relation to keep per epoch (default: 1.0)')
    
    args = parser.parse_args()
    
    # -------- PASS 1: parse only to read --config --------
    pre, _ = parser.parse_known_args()

    config = None
    if pre.config:
        cfg_path = Path(pre.config).expanduser()
        if not cfg_path.is_absolute():
            if not cfg_path.exists():
                cfg_path = REPO_ROOT / pre.config
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")

        config = utils.load_yaml_config(cfg_path)
        config = utils.resolve_config_paths(config, base_dir=cfg_path.parent, repo_root=REPO_ROOT)

        # Feed only argparse-known keys as defaults so CLI can override them
        known_dests = {a.dest for a in parser._actions if a.dest}
        yaml_defaults = {k: v for k, v in config.items() if k in known_dests}

        parser.set_defaults(**yaml_defaults)

    # -------- PASS 2: final parse (CLI > YAML > code defaults) --------
    args = parser.parse_args()

    # If you want to keep extra YAML fields (not in argparse), attach them:
    if config is not None:
        known_dests = {a.dest for a in parser._actions if a.dest}
        args.extra_config = {k: v for k, v in config.items() if k not in known_dests}

    # Always require a dataset CSV
    if args.data_path is None:
        raise ValueError("--data_path is required (no auto-detection).")

    # If a modality is enabled, a path must be provided
    if args.use_visual and args.visual_features_path is None:
        raise ValueError("--use_visual is enabled but --visual_features_path is missing.")
    if args.use_textual and args.textual_features_path is None:
        raise ValueError("--use_textual is enabled but --textual_features_path is missing.")
    if args.use_relation_features and args.relation_features_path is None:
        raise ValueError("--use_relation_features is enabled but --relation_features_path is missing.")

    # --- Expand & validate paths ---
    args.data_path = utils.validate_path(utils.expand_path(args.data_path), "Dataset CSV")
    if args.use_visual:
        args.visual_features_path = utils.validate_path(utils.expand_path(args.visual_features_path), "Visual features")
    if args.use_textual:
        args.textual_features_path = utils.validate_path(utils.expand_path(args.textual_features_path), "Textual features")
    if args.use_relation_features:
        args.relation_features_path = utils.validate_path(utils.expand_path(args.relation_features_path), "Relation features")
    if args.save_path:
        args.save_path = utils.expand_path(args.save_path)

    args.cuda = (not args.no_cuda) and torch.cuda.is_available()
    return args


def main():
    args = parse_args()
    
    # Set seed
    utils.set_seed(args.seed)
    
    # Print configuration
    print(f"\n{'='*80}")
    print(f"VL-KGE Training")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Fusion: {args.fusion_mode}")
    print(f"Modalities:")
    print(f"  Structural: {args.use_structural}")
    print(f"  Visual: {args.use_visual}")
    print(f"  Textual: {args.use_textual}")
    print(f"  Relation: {args.use_relation_features}")
    print(f"Advanced Features:")
    print(f"  Visual projection adapter: {args.visual_proj}")
    print(f"  Textual projection adapter: {args.textual_proj}")
    print(f"  Shared projection: {args.shared_projection}")
    print(f"Dataset Options:")
    print(f"  Per-relation candidates: {args.use_per_relation_candidates}")
    print(f"  Bidirectional eval: {args.bidirectional_eval}")
    print(f"  Inductive learning: {args.inductive}")
    print(f"  Modality asymmetry: {args.modality_asymmetry}")
    print(f"{'='*80}\n")
    
    # Initialize DataLoader
    print("Loading dataset...")
    data_loader = KnowledgeGraphDataLoader(
        data_path=args.data_path,
        dataset_name=args.dataset,
        exclude_relations=args.exclude_relations,
        exclude_relations_eval=args.exclude_relations_eval,
        add_inverse_relations=args.add_inverse_relations,
        use_per_relation_candidates=args.use_per_relation_candidates,
        artist2artist_relations=args.artist2artist_relations,
        bidirectional_eval=args.bidirectional_eval
    )
    
    entity_to_id, relation_to_id = data_loader.get_entities_and_relations()
    
    # Load features using utils
    print("\nLoading features...")
    visual_features, textual_features, relation_features, visual_entity_to_index, textual_entity_to_index = \
        utils.load_features(args, entity_to_id, relation_to_id)
    
    # Split data
    train_data, val_data, test_data = data_loader.split_data()
    split_sizes = data_loader.get_split_sizes()
    print(f"\nData splits: Train={split_sizes['train']:,}, Val={split_sizes['val']:,}, Test={split_sizes['test']:,}")
    
    # Build seen entity set for inductive evaluation
    train_entity_ids = None
    if args.inductive:
        seen_entities = set(train_data['head']).union(set(train_data['tail']))
        train_entity_ids = torch.tensor(
            [entity_to_id[e] for e in seen_entities if e in entity_to_id],
            dtype=torch.long
        )
        print(f"Seen entities (for inductive eval): {len(train_entity_ids):,}")
    
    # Compute filter map and relation probs
    print("\nBuilding evaluation mappings...")
    filter_map = data_loader.compute_filter_map()
    relation_probs = data_loader.compute_relation_probs()
    
    # Create DataLoaders
    generator = torch.Generator().manual_seed(args.seed)
    
    train_dataset = KGDataset(train_data, entity_to_id, relation_to_id)
    val_dataset = KGDataset(val_data, entity_to_id, relation_to_id)
    test_dataset = KGDataset(test_data, entity_to_id, relation_to_id)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Sample train loader for quick evaluation
    sample_frac = 0.1 if args.dataset == 'wn9_img' else 0.01

    # Exclude relations from sampled training evaluation if specified
    exclude_from_sampled_train = args.exclude_relations_eval or []

    # For WikiArt-MKG-v2, also exclude hasCreatedArtwork from sampled eval
    if args.dataset == 'wikiart_mkg_v2' and 'hasCreatedArtwork' not in exclude_from_sampled_train:
        exclude_from_sampled_train = list(exclude_from_sampled_train) + ['hasCreatedArtwork']

    if exclude_from_sampled_train:
        sampled_train_data = train_data[~train_data['relation'].isin(exclude_from_sampled_train)]
        sampled_train_data = sampled_train_data.sample(frac=sample_frac, random_state=args.seed)
        print(f"Excluded {exclude_from_sampled_train} from sampled training evaluation")
    else:
        sampled_train_data = train_data.sample(frac=sample_frac, random_state=args.seed)

    sampled_train_dataset = KGDataset(sampled_train_data, entity_to_id, relation_to_id)
    sampled_train_loader = DataLoader(sampled_train_dataset, batch_size=args.batch_size, 
                                     shuffle=False)
    print(f"Sampled {len(sampled_train_data):,} training triples for quick evaluation")
    
    # Prepare training data for relation downsampling
    train_df_for_downsampling = None
    if args.downsample_relation is not None:
        print(f"\nRelation downsampling enabled for: {args.downsample_relation}")
        print(f"  Keeping {args.downsample_fraction*100:.2f}% per epoch")
        train_df_for_downsampling = train_data
    
    # Initialize device
    device = utils.set_gpu() if args.cuda else torch.device("cpu")
    
    # Create save directory if needed
    if args.save_path:
        save_dir = os.path.dirname(args.save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Created checkpoint directory: {save_dir}")
    
    # Initialize model
    print("\nInitializing model...")
    model, optimizer, scheduler = helpers.get_model(
        model_name=args.model,
        num_entities=len(entity_to_id),
        num_relations=len(relation_to_id),
        visual_features=visual_features,
        textual_features=textual_features,
        relation_features=relation_features,
        visual_entity_to_index=visual_entity_to_index,
        textual_entity_to_index=textual_entity_to_index,
        embedding_dim=args.embedding_dim,
        fusion_mode=args.fusion_mode,
        use_structural=args.use_structural,
        use_visual=args.use_visual,
        use_textual=args.use_textual,
        freeze_visual=args.freeze_visual,
        freeze_textual=args.freeze_textual,
        visual_proj=args.visual_proj,
        textual_proj=args.textual_proj,
        shared_projection=args.shared_projection,
        inductive=args.inductive,
        modality_asymmetry=args.modality_asymmetry,
        normalize_before_fusion=args.normalize_before_fusion,
        # Model-specific arguments
        p_norm=args.p_norm,
        normalize_relations=args.normalize_relations,
        margin=args.margin,
        lr=args.lr,
        use_scheduler=args.use_scheduler,
        device=device
    )
    
    print(f"\n{model}")
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Mark seen entities for inductive evaluation
    if train_entity_ids is not None and hasattr(model, 'mark_seen_entities'):
        model.mark_seen_entities(train_entity_ids)
        print("Marked seen entities for inductive evaluation")
    
    # Train
    helpers.train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        sampled_train_loader=sampled_train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        filter_map=filter_map,
        relation_probs=relation_probs,
        relation_to_valid_tails_train=data_loader.relation_to_valid_tails_train if args.use_per_relation_candidates else None,
        relation_to_valid_tails_val=data_loader.relation_to_valid_tails_eval_val if args.use_per_relation_candidates else None,
        relation_to_valid_tails_test=data_loader.relation_to_valid_tails_eval_test if args.use_per_relation_candidates else None,
        relation_to_id=relation_to_id,
        # Relation downsampling
        train_data=train_df_for_downsampling,
        entity_to_id=entity_to_id,
        downsample_relation=args.downsample_relation,
        downsample_fraction=args.downsample_fraction,
        batch_size=args.batch_size,
        # Training parameters
        num_epochs=args.epochs,
        num_neg_samples=args.num_neg_samples,
        use_bernoulli=args.use_bernoulli,
        top_k=args.top_k,
        dataset_name=args.dataset,
        bidirectional_eval=args.bidirectional_eval,
        device=device,
        save_path=args.save_path,
        generator=generator,
        evaluate_every=args.evaluate_every,
        patience=args.patience,
        resume_from=args.resume_from
    )


if __name__ == '__main__':
    main()