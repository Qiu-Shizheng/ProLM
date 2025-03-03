import pandas as pd
import numpy as np
from proteinbert import load_pretrained_model
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs
import keras.backend as K
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def process_and_save_features(csv_file, output_file, excluded_ids_file, proteinbert_dir, pretrained_model_path,
                              batch_size=128, max_seq_len=16384):
    # Read CSV file
    print(f"Reading protein sequences from {csv_file}...")
    df = pd.read_csv(csv_file)

    ids = []
    seqs = []
    excluded_ids = []

    for index, row in df.iterrows():
        seq_id = row['Protein']
        seq = str(row['Sequence'])

        if len(seq) > max_seq_len:
            excluded_ids.append(seq_id)
            continue

        ids.append(seq_id)
        seqs.append(seq)

    print(f"Number of sequences meeting the length requirement: {len(ids)}")
    print(f"Number of excluded sequences: {len(excluded_ids)}")

    # Save the excluded IDs
    with open(excluded_ids_file, 'w') as f:
        for seq_id in excluded_ids:
            f.write(seq_id + '\n')
    print(f"Excluded IDs saved to {excluded_ids_file}")

    # Load the pretrained ProteinBERT model
    print("Loading the pretrained ProteinBERT model...")
    pretrained_model_generator, input_encoder = load_pretrained_model(
        local_model_dump_dir=proteinbert_dir,
        local_model_dump_file_name=pretrained_model_path
    )

    model = get_model_with_hidden_layers_as_outputs(pretrained_model_generator.create_model(max_seq_len))
    print("Model loaded successfully.")

    # Extract features
    features = {}
    print("Starting feature extraction...")
    for i in tqdm(range(0, len(ids), batch_size), desc="Processing Batches"):
        batch_ids = ids[i:i + batch_size]
        batch_seqs = seqs[i:i + batch_size]

        encoded_x = input_encoder.encode_X(batch_seqs, max_seq_len)
        _, global_representations = model.predict(encoded_x, batch_size=batch_size)

        for j, seq_id in enumerate(batch_ids):
            features[seq_id] = global_representations[j]

    # Save features to a .npz file
    np.savez(output_file, **features)
    print(f"Features saved to {output_file}")


if __name__ == "__main__":
    csv_file = '/protein_sequences.csv'
    output_file = '/path/global_representations.npz'
    excluded_ids_file = '/path/excluded_ids.txt'
    proteinbert_dir = '/path/proteinbert'
    pretrained_model_path = '/path/full_go_epoch_92400_sample_23500000.pkl'

    process_and_save_features(
        csv_file=csv_file,
        output_file=output_file,
        excluded_ids_file=excluded_ids_file,
        proteinbert_dir=proteinbert_dir,
        pretrained_model_path=pretrained_model_path,
        batch_size=128,
        max_seq_len=16384
    )