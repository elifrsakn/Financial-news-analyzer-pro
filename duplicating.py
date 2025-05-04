import pandas as pd

# 📂 1. Final dengeli dataset'i yükleyelim
df = pd.read_csv("/Users/elifsakin/Desktop/dg_hediye/final_balanced_stock_sentiment.csv")

# 🔍 2. Sınıflara ayıralım
positive_df = df[df['Sentiment'] == 'positive']
neutral_df = df[df['Sentiment'] == 'neutral']
negative_df = df[df['Sentiment'] == 'negative']

print(f"✅ Positive örnek sayısı: {positive_df.shape[0]}")
print(f"✅ Neutral örnek sayısı: {neutral_df.shape[0]}")
print(f"✅ Negative örnek sayısı: {negative_df.shape[0]}")

# 🧩 3. Data augmentation (Random sampling ile çoğaltma)

# 1000 positive ve 1000 negative cümle seçelim
positive_augmented = positive_df.sample(1000, replace=True, random_state=42)
negative_augmented = negative_df.sample(1000, replace=True, random_state=42)

print(f"✅ 1000 Positive ve 1000 Negative örnek augment edildi!")

# 🧩 4. Hepsini birleştirelim
augmented_df = pd.concat([df, positive_augmented, negative_augmented], ignore_index=True)

# 🌀 5. Shuffle yapalım (karıştıralım)
augmented_df = augmented_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 📈 6. Yeni sınıf dağılımı
print("\n✅ Augmented Dataset Şekli:", augmented_df.shape)
print(augmented_df['Sentiment'].value_counts())

# 📥 7. Yeni dataset'i CSV olarak kaydedelim
augmented_df.to_csv("/Users/elifsakin/Desktop/dg_hediye/final_augmented_stock_sentiment.csv", index=False)

print("\n✅ Final augmented dataset başarıyla kaydedildi: /Users/elifsakin/Desktop/dg_hediye/final_augmented_stock_sentiment.csv")
