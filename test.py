import os
import pickle
import argparse
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from train import BertDataset
from eval import evaluate
from model.contrast import ContrastModel

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--batch', type=int, default=32, help='Batch size.')
parser.add_argument('--data', type=str, default='WOS-150-H2', help='Dataset.')
parser.add_argument('--fold', type=int, default=0, help='Fold index for cross-validation.')
args = parser.parse_args()

if __name__ == '__main__':

    checkpoint = torch.load(f"resource/model_checkpoint/HGCLR_{args.data}/HGCLR_{args.data}_{args.fold}.pt", map_location='cpu')

    batch_size = args.batch
    device = args.device
    fold = args.fold

    # Retrieve the original arguments used during training
    train_args = checkpoint['args'] if checkpoint['args'] is not None else args

    # Point data_path to the correct resource directory
    data_path = os.path.join('resource', 'dataset', train_args.data)

    if not hasattr(train_args, 'graph'):
        train_args.graph = False
    print(train_args)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    label_dict = torch.load(os.path.join(data_path, 'bert_value_dict.pt'))
    label_dict = {i: tokenizer.decode(v, skip_special_tokens=True) for i, v in label_dict.items()}
    num_class = len(label_dict)

    dataset = BertDataset(device=device, pad_idx=tokenizer.pad_token_id, data_path=data_path)
    model = ContrastModel.from_pretrained('bert-base-uncased', num_labels=num_class,
                                          contrast_loss=train_args.contrast, graph=train_args.graph,
                                          layer=train_args.layer, data_path=data_path, multi_label=train_args.multi,
                                          lamb=train_args.lamb, threshold=train_args.thre)

    # 2. Load the specific split for this fold and cast it to a list
    split = torch.load(os.path.join(data_path, f'split_fold_{fold}.pt'))
    test_subset = Subset(dataset, list(split['test']))
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    print(f"Testing with: {len(test_subset)} samples")

    model.load_state_dict(checkpoint['param'])
    model.to(device)

    truth = []
    pred = []

    # 3. Initialize the dictionary for the ranked predictions
    ranking = {}

    model.eval()
    pbar = tqdm(test_loader)
    with torch.no_grad():
        for data, label, batch_idx in pbar:
            padding_mask = data != tokenizer.pad_token_id
            output = model(data, padding_mask, return_dict=True)

            # Extract probabilities
            batch_scores = torch.sigmoid(output['logits']).cpu().tolist()

            # Format ground truth for baseline evaluation
            for l in label:
                t = []
                for i in range(l.size(0)):
                    if l[i].item() == 1:
                        t.append(i)
                truth.append(t)

            # Collect scores for baseline evaluation
            for scores in batch_scores:
                pred.append(scores)

            # 4. Construct the ranking dictionary per document
            for i, text_id in enumerate(batch_idx):
                text_key = f"text_{text_id}"
                ranking[text_key] = {}
                for label_id, score in enumerate(batch_scores[i]):
                    ranking[text_key][f"label_{label_id}"] = score

    pbar.close()

    # Standard baseline evaluation
    scores = evaluate(pred, truth, label_dict)
    macro_f1 = scores['macro_f1']
    micro_f1 = scores['micro_f1']
    print('macro', macro_f1, 'micro', micro_f1)

    # 5. Export the ranking dictionary to a Pickle file
    ranking_dir = os.path.join('resource', 'ranking', f'HGCLR_{args.data}')
    os.makedirs(ranking_dir, exist_ok=True)

    # Using the dynamically generated name
    ranking_file = os.path.join(ranking_dir, f"HGCLR_{args.data}_{args.fold}.rnk")

    # Opening the file in binary write mode ('wb') for pickle
    with open(ranking_file, 'wb') as f:
        pickle.dump(ranking, f)

    print(f"Ranking successfully saved to: {ranking_file}")