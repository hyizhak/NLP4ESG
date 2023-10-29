import os
import warnings
import pandas as pd
import torch
import re
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
import torch.nn.utils.rnn as rnn_utils
from tqdm import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)


class ReportsDataset(Dataset):
    def __init__(self, data=None):
        if data is not None:
            self.df = data
            return
        else:
            self.df = pd.DataFrame(columns=['Code', 'year', 'sentences', 'embeddings'])  # Create an empty DataFrame
            reports_dir = 'reports'
            files = [f for f in os.listdir(reports_dir) if os.path.isfile(os.path.join(reports_dir, f))]
            texts = []

            for file in tqdm(files, desc="Reading reports"):
                # Parsing file name for code and year
                code, year_with_extension = file.split('_')
                if year_with_extension == 'Store':
                    continue
                year = int(year_with_extension.replace('.txt', ''))

                # Only process files from 2011 to 2021
                if 2009 <= year <= 2022:
                    with open(os.path.join(reports_dir, file), 'r') as f:
                        texts.append(f.read())

            self.sentences_per_report = []

            for text in tqdm(texts, desc="Splitting sentences"):
                sentences = re.split(r'[。！？]', text)
                sentences = [s.replace(r'\s+', "") for s in sentences if s]
                self.sentences_per_report.append(sentences)

            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

            self.embeddings_per_report = []

            for i, sentences in tqdm(enumerate(self.sentences_per_report), desc="Embedding sentences"):
                embeddings = model.encode(sentences)
                self.embeddings_per_report.append(embeddings)

                # Parsing file name for code and year
                code, year = files[i].split('_')
                year = year.replace('.txt', '')

                # Adding data to DataFrame
                self.df = self.df.append({'Code': code, 'year': year, 'sentences': sentences,
                                          'embeddings': embeddings}, ignore_index=True)

            self.padded_embeddings = rnn_utils.pad_sequence([torch.tensor(report) for report in
                                                             self.embeddings_per_report], batch_first=True,
                                                            padding_value=0)

            self.df['embeddings'] = [report for report in self.padded_embeddings]

            excel_data = pd.read_excel('esgrating/chiindex09-22.xlsx', dtype={'Code': str})

            excel_data.drop(columns=['证券简称', '上市日期', '所属证监会行业名称'], axis=1, inplace=True)

            esg_mapping = {'AAA': 8, 'AA': 7, 'A': 6, 'BBB': 5, 'BB': 4, 'B': 3, 'CCC': 2, 'CC': 1, 'C': 0}

            excel_data['Code'] = excel_data['Code'].str[:6]

            excel_melted = excel_data.melt(id_vars='Code', var_name='year', value_name='ESG')

            excel_melted['year'] = excel_melted['year'].astype(str)

            excel_melted['ESG'] = excel_melted['ESG'].map(esg_mapping)

            print(excel_melted.head())

            self.df = pd.merge(self.df, excel_melted, on=['Code', 'year'], how='inner')

            self.df.reset_index(drop=True, inplace=True)

            self.df.to_pickle('full_data.pkl')

            print(self.df.head())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx]['embeddings'], self.df.iloc[idx]['ESG']


# test = ReportsDataset()

