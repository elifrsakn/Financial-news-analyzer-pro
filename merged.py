import pandas as pd

# 📂 1. Ana Dataset (senin eski çalıştığın dataset)
main_df = pd.read_csv("/Users/elifsakin/Desktop/dg_hediye/data.csv")

# 📂 2. Uploaded Dataset (şu anda yüklediğin all-data.csv dosyası)
uploaded_df = pd.read_csv("/Users/elifsakin/Desktop/dg_hediye/all-data.csv", encoding="ISO-8859-1", header=None)
uploaded_df.columns = ['Sentiment', 'Sentence']

# 🔍 3. Negative ve Positive örnekleri ayrı ayrı seçelim
negative_samples = uploaded_df[uploaded_df['Sentiment'] == 'negative']
positive_samples = uploaded_df[uploaded_df['Sentiment'] == 'positive']

print(f"✅ Seçilen yeni negatif örnek sayısı: {negative_samples.shape[0]}")
print(f"✅ Seçilen yeni pozitif örnek sayısı: {positive_samples.shape[0]}")

# 🧩 4. Ana dataset + negative örnekler + positive örnekler ➔ Hepsini birleştirelim
combined_df = pd.concat([main_df, negative_samples, positive_samples], ignore_index=True)

# 🎯 5. İlk birleşim sonrası sınıf dağılımını görelim
print("\n✅ İlk birleşim sonrası dataset şekli:", combined_df.shape)
print(combined_df['Sentiment'].value_counts())

# 📥 6. Geçici olarak kaydedelim (istersen)
combined_df.to_csv("/Users/elifsakin/Desktop/dg_hediye/final_enhanced_stock_sentiment.csv", index=False)

print("\n✅ Final enhanced dataset başarıyla kaydedildi: /Users/elifsakin/Desktop/dg_hediye/final_enhanced_stock_sentiment.csv")

# ------------------------------------------
# 🧠 7. NEGATIVE SINIFI ÇOĞALTMA (balancing)

# 📂 Şimdi final_enhanced_stock_sentiment.csv dosyasını yeniden yükleyelim
enhanced_df = pd.read_csv("/Users/elifsakin/Desktop/dg_hediye/final_enhanced_stock_sentiment.csv")

# Sınıfları ayıralım
positive_df = enhanced_df[enhanced_df['Sentiment'] == 'positive']
neutral_df = enhanced_df[enhanced_df['Sentiment'] == 'neutral']
negative_df = enhanced_df[enhanced_df['Sentiment'] == 'negative']

# Target: negative sınıfı positive ve neutral ile dengelemek (~3100-3200 civarı)

target_negative_count = 3100
multiplier = target_negative_count // negative_df.shape[0] + 1  # Kaç kat çoğaltacağız

# Negative cümleleri çoğalt
negative_df_augmented = pd.concat([negative_df] * multiplier, ignore_index=True)

# Fazla çoğalmışsa, hedef sayıya kırpıyoruz
negative_df_augmented = negative_df_augmented.sample(target_negative_count, random_state=42)

# 🎯 Hepsini birleştirip karıştıralım
final_balanced_df = pd.concat([positive_df, neutral_df, negative_df_augmented], ignore_index=True)
final_balanced_df = final_balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 📈 Final sınıf dağılımı
print("\n✅ Final dengelenmiş dataset şekli:", final_balanced_df.shape)
print(final_balanced_df['Sentiment'].value_counts())

# 📥 8. Final balanced dataset'i kaydedelim
final_balanced_df.to_csv("/Users/elifsakin/Desktop/dg_hediye/final_balanced_stock_sentiment.csv", index=False)

print("\n✅ Final balanced dataset başarıyla kaydedildi: /Users/elifsakin/Desktop/dg_hediye/final_balanced_stock_sentiment.csv")
