#!/usr/bin/env python3
"""
Demo script showing rerun visualization of EuRoC datasets
Equivalent to running MATLAB dataset_plot functionality

Usage:
    python demo_plot.py                          # Default: MH_01_easy, 30 seconds
    python demo_plot.py MH_02_easy              # Specific dataset, 30 seconds
    python demo_plot.py V1_01_easy 60           # Vicon dataset, 60 seconds
    python demo_plot.py --help                  # Show options
"""

import sys
import argparse
from pathlib import Path

from dataset_loader import dataset_load, print_dataset_summary
from dataset_plot import dataset_plot

# Available EuRoC datasets in standard location
AVAILABLE_DATASETS = {
    'MH_01_easy': '../../EuRoc_ASL/MH_01_easy',
    'MH_02_easy': '../../EuRoc_ASL/MH_02_easy', 
    'MH_03_medium': '../../EuRoc_ASL/MH_03_medium',
    'MH_04_difficult': '../../EuRoc_ASL/MH_04_difficult',
    'MH_05_difficult': '../../EuRoc_ASL/MH_05_difficult',
    'V1_01_easy': '../../EuRoc_ASL/V1_01_easy',
    'V1_02_medium': '../../EuRoc_ASL/V1_02_medium',
    'V1_03_difficult': '../../EuRoc_ASL/V1_03_difficult',
    'V2_01_easy': '../../EuRoc_ASL/V2_01_easy',
    'V2_02_medium': '../../EuRoc_ASL/V2_02_medium',
    'V2_03_difficult': '../../EuRoc_ASL/V2_03_difficult',
}


def main():
    parser = argparse.ArgumentParser(
        description='Visualize EuRoC datasets using rerun (equivalent to MATLAB dataset_plot)',
        epilog=f'Available datasets: {", ".join(AVAILABLE_DATASETS.keys())}'
    )
    
    parser.add_argument(
        'dataset', 
        nargs='?', 
        default='MH_01_easy',
        help='Dataset name or path (default: MH_01_easy)'
    )
    
    parser.add_argument(
        'max_time', 
        nargs='?', 
        type=float, 
        default=30.0,
        help='Maximum time to visualize in seconds (default: 30.0)'
    )
    
    parser.add_argument(
        '--list-datasets', 
        action='store_true',
        help='List available datasets and exit'
    )
    
    parser.add_argument(
        '--no-spawn', 
        action='store_true',
        help='Don\'t automatically open rerun viewer'
    )
    
    parser.add_argument(
        '--summary', 
        action='store_true',
        help='Print dataset summary before visualization'
    )
    
    args = parser.parse_args()
    
    # List datasets and exit
    if args.list_datasets:
        print("Available EuRoC datasets:")
        for name, path in AVAILABLE_DATASETS.items():
            exists = "✓" if Path(path).exists() else "✗"
            print(f"  {exists} {name:<15} -> {path}")
        return
    
    # Resolve dataset path
    if args.dataset in AVAILABLE_DATASETS:
        dataset_path = AVAILABLE_DATASETS[args.dataset]
        dataset_name = args.dataset
    else:
        dataset_path = args.dataset
        dataset_name = Path(dataset_path).name
    
    # Check if dataset exists
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("\nTip: Use --list-datasets to see available datasets")
        sys.exit(1)
    
    print("="*60)
    print(f"🚀 EuRoC Dataset Visualization with Rerun")
    print(f"📂 Dataset: {dataset_name}")
    print(f"⏱️  Duration: {args.max_time}s")
    print(f"🎯 Path: {dataset_path}")
    print("="*60)
    
    try:
        # Load dataset
        print("\n📊 Loading dataset...")
        dataset = dataset_load(dataset_path)
        
        # Print summary if requested
        if args.summary:
            print_dataset_summary(dataset)
        
        # Launch visualization
        print("\n🎨 Starting rerun visualization...")
        print("   This will:")
        print("   • Show 3D sensor configuration and coordinate frames")
        print("   • Plot 3D trajectory with pose markers")
        print("   • Display IMU time series (gyroscope + accelerometer)")
        print("   • Visualize camera calibration data (if available)")
        print()
        
        dataset_plot(
            dataset,
            dataset_path=dataset_path,
            spawn_viewer=not args.no_spawn,
            max_time_sec=args.max_time
        )
        
        print("\n✅ Visualization complete!")
        
        if not args.no_spawn:
            print("🌐 Rerun viewer should open automatically")
            print("   If not, visit: http://localhost:9876")
        else:
            print("🌐 Start rerun viewer with: rerun")
        
        print("\n💡 Tips:")
        print("   • Use mouse to navigate the 3D scene")
        print("   • Click timeline to scrub through trajectory")
        print("   • Toggle sensor data visibility in the left panel")
        print("   • Zoom/pan time series plots on the right")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 