import os
from datetime import datetime


def setup_logging(save_path, run_id, args):
    """Setup comprehensive logging system"""
    train_log_path = os.path.join(save_path, f"training_log_{run_id}.txt")
    sam_log_path = os.path.join(save_path, f"sam_training_log_{run_id}.txt")
    refine_log_path = os.path.join(save_path, f"refine_training_log_{run_id}.txt")
    
    with open(train_log_path, "w", encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Training Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Task Name: {args.task_name}\n")
        f.write(f"Model Type: {args.model_type}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"SAM Training Epochs: {args.num_epochs_sam}\n")
        f.write(f"Refine Training Epochs: {args.num_epochs_refine}\n")
        f.write(f"Evaluation Metrics: {', '.join(args.metrics)}\n")
        f.write("=" * 80 + "\n\n")
    
    with open(sam_log_path, "w", encoding='utf-8') as f:
        f.write("SAM Model Training Log\n")
        f.write("=" * 50 + "\n")
        f.write(f"Run ID: {run_id}\n")
        f.write("Epoch\tTime\t\tLR\t\tTrain_Loss\tVal_Loss\t")
        for metric in args.metrics:
            f.write(f"Train_{metric.upper()}\tVal_{metric.upper()}\t")
        f.write("\n")
    
    with open(refine_log_path, "w", encoding='utf-8') as f:
        f.write("Refinement Network Training Log\n")
        f.write("=" * 50 + "\n")
        f.write(f"Run ID: {run_id}\n")
        f.write("Epoch\tTime\t\tLR\t\tTrain_Loss\tVal_Loss\t")
        for metric in args.metrics:
            f.write(f"Train_{metric.upper()}\tVal_{metric.upper()}\t")
        f.write("\n")
    
    return train_log_path, sam_log_path, refine_log_path


def log_epoch_results(log_path, epoch, current_lr, train_loss, val_loss, train_metrics, val_metrics, args):
    """Log training results for each epoch"""
    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    
    with open(log_path, "a", encoding='utf-8') as f:
        f.write(f"{epoch}\t{current_time}\t{current_lr:.6f}\t{train_loss:.6f}\t{val_loss:.6f}\t")
        for metric in args.metrics:
            train_val = train_metrics.get(metric, 0.0)
            val_val = val_metrics.get(metric, 0.0)
            f.write(f"{train_val:.4f}\t{val_val:.4f}\t")
        f.write("\n")


def log_training_summary(train_log_path, stage_name, best_metrics, total_epochs):
    """Log training stage summary"""
    with open(train_log_path, "a", encoding='utf-8') as f:
        f.write(f"\n{stage_name} Training Completed\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Training Epochs: {total_epochs}\n")
        f.write("Best Validation Metrics:\n")
        for metric, value in best_metrics.items():
            f.write(f"  {metric.upper()}: {value:.4f}\n")
        f.write(f"Completion Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-" * 40 + "\n\n")


def log_final_summary(train_log_path, model_save_path):
    """Log final training summary"""
    with open(train_log_path, "a", encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Training Completed Successfully\n")
        f.write("=" * 80 + "\n")
        f.write(f"Final Completion Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Model Save Path: " + model_save_path + "\n")
        f.write("=" * 80 + "\n")


def print_training_info(epoch, train_epoch_loss, val_epoch_loss, train_metrics, val_metrics):
    """Print formatted training information to console"""
    train_metrics_str = {k: f'{v:.4f}' for k, v in train_metrics.items()}
    val_metrics_str = {k: f'{v:.4f}' for k, v in val_metrics.items()}
    print(f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, '
          f'Train_loss: {train_epoch_loss:.6f}, Train metrics: {train_metrics_str}, '
          f'Val_loss:{val_epoch_loss:.6f}, Val metrics: {val_metrics_str}')
