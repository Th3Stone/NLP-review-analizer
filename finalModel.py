import os
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
import re 
import string
from collections import Counter
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import gradio as gr


def load_local_data(dataset_path):
    
    reviews, labels, ratings, domains_list = [], [], [], []
    
    # The 4 domain folders in the dataset
    domains = ['books', 'dvd', 'electronics', 'kitchen_&_housewares']
    
    # We map reviews
    file_types = {'positive.review': 1, 'negative.review': 0}
    
    print(f"Looking for data in: '{dataset_path}'...")
    print("Parsing pseudo-XML files...")
    
    for domain in domains:
        for f_name, label in file_types.items():
            # Build the path to the specific file
            file_path = os.path.join(dataset_path, domain, f_name)
            
            if os.path.exists(file_path):
                # Using latin-1 encoding as these are older web-scraped texts
                with open(file_path, 'r', encoding='latin-1') as file:
                    content = file.read()
                    
                    # Parse the pseudo-XML 
                    soup = BeautifulSoup(content, 'lxml')
                    
                    # Find all <review> blocks
                    for review_tag in soup.find_all('review'):
                        # Extract title and text
                        title_tag = review_tag.find('title')
                        text_tag = review_tag.find('review_text')
                        
                        title = title_tag.text.strip() if title_tag else ""
                        text = text_tag.text.strip() if text_tag else ""
                        
                        # Combine title and text for richer sentiment data
                        full_review = f"{title}. {text}".strip()
                        
                        # Extract the star rating (for our ethical analysis later)
                        rating_tag = review_tag.find('rating')
                        try:
                            rating = float(rating_tag.text.strip()) if rating_tag else -1.0
                        except ValueError:
                            rating = -1.0
                            
                        # Only append if we actually have a reasonable amount of text
                        if len(full_review) > 5: 
                            reviews.append(full_review)
                            labels.append(label)
                            ratings.append(rating)
                            domains_list.append(domain)
            else:
                print(f"Warning: Could not find {file_path}")
                            
    # Create a DataFrame for easier manipulation in Pandas
    df = pd.DataFrame({
        'review_text': reviews, 
        'label': labels, 
        'original_rating': ratings,
        'domain': domains_list
    })
    
    # Shuffle the dataset to randomize the domains and sentiments
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

# Run the data loader with your exact folder path
dataset_directory = "/home/josepiedrahita/Downloads/a3/sorted_data_acl"
df = load_local_data(dataset_path=dataset_directory)

print(f"\nTotal reviews loaded: {len(df)}")
print("\nSample data:")
print(df[['original_rating', 'label', 'review_text']].head())

#---------SECTION 2-----------------
def clean_text(text):
   
    text = text.lower()

    text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)
 
    text = re.sub(r"\d+", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text

print(f"Original dataset size: {len(df)}")


# APplying ETHICAL RESPONSIBILITY & contradictory  REVIEWS

# A review labelled positive (1) shouldn't have a 1 or 2 star rating.
# A review labelled negative (0) shouldn't have a 4 or 5 star rating.
# A 3-star rating is neutral and forcing it into binary sentiment introduces bias.

valid_positive = (df['label'] == 1) & (df['original_rating'] >= 4.0)
valid_negative = (df['label'] == 0) & (df['original_rating'] <= 2.0)
missing_rating = (df['original_rating'] == -1.0) # Keep if rating was missing but label existed

df_cleaned = df[valid_positive | valid_negative | missing_rating].copy()

dropped_ethical = len(df) - len(df_cleaned)
print(f"Dropped {dropped_ethical} reviews due to neutral (3-star) or mismatched ratings.")


#TEXT CLEANING
print("Cleaning punctuation and standardising text...")
df_cleaned['clean_text'] = df_cleaned['review_text'].apply(clean_text)


# OUTLIER REMOVAL (Too short review)
df_cleaned['word_count'] = df_cleaned['clean_text'].apply(lambda x: len(x.split()))

df_cleaned = df_cleaned[df_cleaned['word_count'] >= 4]

dropped_length = len(df) - dropped_ethical - len(df_cleaned)
print(f"Dropped {dropped_length} reviews that were too short (outliers).")

# MIX AND RANDOMISE DATA

df_cleaned = df_cleaned.sample(frac=1, random_state=101).reset_index(drop=True)

#results: 
print(f"\nFinal cleaned dataset size: {len(df_cleaned)}")
print("\nSample of cleaned data:")
print(df_cleaned[['label', 'original_rating', 'clean_text']].head())

# No data was filtered, apparently all data was correct

#-------section 3-------------

#tokenize reviews
reviews_split = [text.split() for text in df_cleaned['clean_text']]

all_words = [word for review in reviews_split for word in review]
word_counts = Counter(all_words)

#filtering typos and wierd worlds 
VOCAB_SIZE = 10000

#map each word
common_words = word_counts.most_common(VOCAB_SIZE)
word2idx = {word: i + 2 for i, (word, count) in enumerate(common_words)}

#special tokens
word2idx['<PAD>'] = 0  # Used for padding short reviews
word2idx['<UNK>'] = 1  # Used for "Unknown" words (words not in top 10,000)

print(f"Vocabulary size: {len(word2idx)} words (including PAD and UNK)")

#Encode the words

encoded_reviews = []
for review in reviews_split:
    encoded_review = [word2idx.get(word, word2idx['<UNK>']) for word in review]
    encoded_reviews.append(encoded_review)


#pad remaining data

SEQ_LENGTH = 200
print(f"Applying PRE-PADDING to all sequences to {SEQ_LENGTH} words...")

features = np.zeros((len(encoded_reviews), SEQ_LENGTH), dtype=int)

for i, review in enumerate(encoded_reviews):
    review_len = len(review)
    
    if review_len <= SEQ_LENGTH:
        # PRE-PADDING: Put the review at the END of the zeros
        features[i, SEQ_LENGTH - review_len:] = np.array(review)
    else:
        # If longer than 200, truncate it
        features[i, :] = np.array(review[:SEQ_LENGTH])

#encode labels to a numpy array
labels = np.array(df_cleaned['label'].values)

#results
print(f"Features matrix shape: {features.shape} (Reviews x Sequence Length)")
print(f"Labels array shape: {labels.shape}")
print(f"\n encoded, padded review (first 20 integers):")
print(features[0][:20])

#---------section 4 ----------

# We will do an 80% / 10% / 10% split
split_idx = int(len(features) * 0.8)
train_x, remaining_x = features[:split_idx], features[split_idx:]
train_y, remaining_y = labels[:split_idx], labels[split_idx:]

# Split the Remaining 20% in half to get Validation (10%) and Testing (10%)
test_idx = int(len(remaining_x) * 0.5)
val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

print(f"Training set size:   {len(train_x)}")
print(f"Validation set size: {len(val_x)}")
print(f"Testing set size:    {len(test_x)}")

#pyTORCH datasets creation

train_data = TensorDataset(torch.from_numpy(train_x).long(), torch.from_numpy(train_y).float())
val_data   = TensorDataset(torch.from_numpy(val_x).long(), torch.from_numpy(val_y).float())
test_data  = TensorDataset(torch.from_numpy(test_x).long(), torch.from_numpy(test_y).float())

#obtein batches
batch_size = 50 
train_data = TensorDataset(torch.from_numpy(train_x).long(), torch.from_numpy(train_y).float())
val_data   = TensorDataset(torch.from_numpy(val_x).long(), torch.from_numpy(val_y).float())
test_data  = TensorDataset(torch.from_numpy(test_x).long(), torch.from_numpy(test_y).float())

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
val_loader   = DataLoader(val_data, shuffle=True, batch_size=batch_size)
test_loader  = DataLoader(test_data, shuffle=True, batch_size=batch_size)
print("Data padded and batched successfully!")

#confirm if it works
dataiter = iter(train_loader)
sample_x, sample_y = next(dataiter)

print(f"\nSample batch features shape: {sample_x.shape} (Batch Size x Sequence Length)")
print(f"Sample batch labels shape: {sample_y.shape} (Batch Size)")


#-----section 5--


#netwrok architecture

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        
        super(SentimentLSTM, self).__init__()
        
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # integer words into feature vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                            dropout=drop_prob, batch_first=True)
        
        # prevent overfitting
        self.dropout = nn.Dropout(drop_prob)
        
        self.fc = nn.Linear(hidden_dim, output_size)
        
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        batch_size = x.size(0)
        
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds)
        
        # Get the final output
        lstm_out = lstm_out[:, -1, :] 
        
        out = self.dropout(lstm_out)
        out = self.fc(out)
        sig_out = self.sig(out)
        
        return sig_out
    
model = SentimentLSTM(len(word2idx), 1, 400, 256, 2)




# INSTANTIATE THE NETWORK

# Smaller hyperparameters to prevent overfitting
embedding_dim = 128        # Reduced from 400
hidden_dim = 128           # Reduced from 256
n_layers = 2               

print("Instantiating the rearchitected network...")
model = SentimentLSTM(len(word2idx), 1, embedding_dim, hidden_dim, n_layers)

print("Rearchitected model ready!")
print(model)


#-- section 6 ---------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 5 # Cahnge to 5 epochs isntead of 4

print(f"Starting RE-TRAINING on device: {device}...")
print("-" * 50)

for e in range(epochs):
    model.train() 
    train_loss = 0.0
    train_correct = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output.squeeze(), labels)
        loss.backward()
        
        # Gradient Clipping stabilizes LSTM training (so we dont get the same mistakes as in the first model we tried)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        
        optimizer.step()
        
        train_loss += loss.item()
        predictions = torch.round(output.squeeze()) 
        correct_tensor = predictions.eq(labels.float().view_as(predictions))
        train_correct += torch.sum(correct_tensor).item()
            
    model.eval() 
    val_loss = 0.0
    val_correct = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            loss = criterion(output.squeeze(), labels)
            val_loss += loss.item()
            
            predictions = torch.round(output.squeeze())
            correct_tensor = predictions.eq(labels.float().view_as(predictions))
            val_correct += torch.sum(correct_tensor).item()
            
    train_loss_avg = train_loss / len(train_loader)
    train_acc = train_correct / len(train_loader.dataset)
    val_loss_avg = val_loss / len(val_loader)
    val_acc = val_correct / len(val_loader.dataset)
    
    print(f"Epoch: {e+1}/{epochs}")
    print(f"Train Loss: {train_loss_avg:.4f} | Train Acc: {train_acc*100:.2f}%")
    print(f"Val Loss:   {val_loss_avg:.4f} | Val Acc:   {val_acc*100:.2f}%")
    print("-" * 50)

print("Training Complete!")



#------testing THE MODEL-------------------


model.eval()
test_loss = 0.0
test_correct = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model(inputs)
        loss = criterion(output.squeeze(), labels)
        test_loss += loss.item()
        
        predictions = torch.round(output.squeeze())
        correct_tensor = predictions.eq(labels.float().view_as(predictions))
        test_correct += torch.sum(correct_tensor).item()

test_acc = test_correct / len(test_loader.dataset)
print(f"Test Accuracy: {test_acc*100:.2f}%\n")


#FUNCTION FOR NEW INPUTS

def predict_sentiment(review_text):
   
    model.eval()
    
    # Clean text (using your clean_text function from earlier)
    cleaned_text = clean_text(review_text)
    
    #Tokenize and encode
    words = cleaned_text.split()
    encoded_words = [word2idx.get(word, word2idx['<UNK>']) for word in words]
    
    #Apply PRE-PADDING (Match the rearchitected model)
    if len(encoded_words) <= SEQ_LENGTH:
        # Put zeros at the beginning, words at the end
        padded_words = [word2idx['<PAD>']] * (SEQ_LENGTH - len(encoded_words)) + encoded_words
    else:
        padded_words = encoded_words[:SEQ_LENGTH]
        
    # Convert to tensor
    feature_tensor = torch.tensor([padded_words]).to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(feature_tensor)
        prediction_score = output.squeeze().item()
        
 
    if prediction_score >= 0.5:
        return f"Positive review (Confidence Score: {prediction_score:.4f})"
    else:
        return f"Negative review (Confidence Score: {prediction_score:.4f})"

#CUSTOM INPUTS TEST

print("--- Custom Inference Tests ---")

# Test 1: Clear Positive (In-domain context)
test_pos = "I absolutely love this blender! It works perfectly and saves me so much time."
print(f"Input: '{test_pos}'")
print(f"Output: {predict_sentiment(test_pos)}\n")

# Test 2: Clear Negative (In-domain context)
test_neg = "Terrible product. It arrived broken and the customer service was extremely rude. Do not buy."
print(f"Input: '{test_neg}'")
print(f"Output: {predict_sentiment(test_neg)}\n")

# Test 3: Outside the training data (Software review / modern slang)
test_out = "The UI is super laggy and the new update bricked my phone, totally mid experience ngl."
print(f"Input: '{test_out}'")
print(f"Output: {predict_sentiment(test_out)}\n")


#------ INTERFACE--------
def web_inference(text):
    """
    This function connects the web page input to our PyTorch model.
    """
    if not text.strip():
        return "Please enter a sentence."
        
    # Get the raw prediction from our model
    raw_result = predict_sentiment(text)
    
    # "Positive review" or "Negative review" as text output.
    if "Positive" in raw_result:
        return "Positive review"
    else:
        return "Negative review"

print("Launching web interface...")

#  web page layout
app = gr.Interface(
    fn=web_inference, 
    inputs=gr.Textbox(lines=4, placeholder="Type your review here...", label="Input Sentence"), 
    outputs=gr.Textbox(label="Sentiment Output"),
    title="Product Review Sentiment Analyzer",
    description="Enter a sentence to determine if it is a Positive or Negative review.",
    flagging_mode="never" 
)

# this will provide a local URL to click
app.launch(share=False)



