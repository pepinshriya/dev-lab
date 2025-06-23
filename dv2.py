import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import string

df = pd.read_csv("spam.csv", encoding='latin-1')

print("\n--- HEAD ---\n", df.head())
print("\n--- INFO ---\n")
print(df.info())
print("\n--- DESCRIBE ---\n", df.describe())
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

print("\n--- NULL VALUES ---\n", df.isnull().sum())

print("\n--- CLASS DISTRIBUTION ---\n", df['label'].value_counts())
sns.countplot(x='label', data=df)
plt.title("Class Distribution")
plt.show()

df['message_length'] = df['message'].apply(len)
sns.histplot(data=df, x='message_length', hue='label', bins=50, kde=True)
plt.title("Message Length Distribution by Label")
plt.show()

from collections import Counter
def get_words(text):
    words = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
    return words

ham_words = []
spam_words = []

for msg, label in zip(df['message'], df['label']):
    if label == 'ham':
        ham_words += get_words(msg)
    else:
        spam_words += get_words(msg)

print("\nTop 10 Ham Words:\n", Counter(ham_words).most_common(10))
print("\nTop 10 Spam Words:\n", Counter(spam_words).most_common(10))

ham_text = ' '.join(ham_words)
spam_text = ' '.join(spam_words)

ham_wc = WordCloud(width=600, height=400, background_color='white').generate(ham_text)
spam_wc = WordCloud(width=600, height=400, background_color='white').generate(spam_text)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(ham_wc, interpolation='bilinear')
plt.axis('off')
plt.title("Ham Word Cloud")

plt.subplot(1, 2, 2)
plt.imshow(spam_wc, interpolation='bilinear')
plt.axis('off')
plt.title("Spam Word Cloud")

plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x='label', y='message_length', data=df)
plt.title("Message Length vs Label")
plt.show()
print("\nEDA Completed. Key Insights:")
print("Ham messages are more frequent than spam.")
print("Spam messages tend to be longer.")
print("Certain keywords appear frequently in spam messages.")
