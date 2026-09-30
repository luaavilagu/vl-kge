import argparse
from pathlib import Path

import torch

from vlkge import helpers, utils
from vlkge.dataloader import KnowledgeGraphDataLoader
from vlkge.mas.environment import KnowledgeGraphEnvironment
from vlkge.mas.evidence_adapter import EvidenceAdapter

try:
    from argparse import BooleanOptionalAction
except ImportError:
    class BooleanOptionalAction(argparse.Action):
        def __init__(self, option_strings, dest, default=None,
                     required=False, help=None):
            option_list = []
            for option in option_strings:
                option_list.append(option)
                if option.startswith("--"):
                    option_list.append("--no-" + option[2:])
            super().__init__(
                option_strings=option_list,
                dest=dest,
                nargs=0,
                default=default,
                required=required,
                help=help,
            )

        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, not option_string.startswith("--no-"))

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent.parent

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

    parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to a trained MASNeuralTransE checkpoint",
    )
    parser.add_argument("--head", type=str, required=True)
    parser.add_argument("--relation", type=str, required=True)
    parser.add_argument("--tail", type=str, required=True)
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='wn9_img',
                       help='Dataset name (default: wn9_img)')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to the dataset CSV (auto-detected for built-in datasets)')
    
    # Model
    parser.add_argument('--model', type=str, default='TransE',
                       choices=['TransE', 'DistMult', 'ComplEx', 'RotatE', 'NeuralTransE', 'MASNeuralTransE'],
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
    device = utils.set_gpu() if args.cuda else torch.device("cpu")

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

    # Initialize model
    print("\nInitializing model...")
    model, optimizer, scheduler = helpers.get_model(
        model_name="MASNeuralTransE",
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

    checkpoint_path = utils.validate_path(
        utils.expand_path(args.checkpoint),
        "MAS checkpoint",
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'model_state_dict'.")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    adapter = EvidenceAdapter(
        model=model,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
        device=device,
    )

    environment = KnowledgeGraphEnvironment(
        visual_evidence_model=adapter.score_visual,
        textual_evidence_model=adapter.score_textual,
        seed=args.seed,
    )

    candidate = {
        "head": args.head,
        "relation": args.relation,
        "tail": args.tail,
    }

    environment.publish_candidate(candidate)
    environment.step()

    for message in environment.messages:
        print(message)

if __name__ == "__main__":
    main()
