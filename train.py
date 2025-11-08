"""
Hindi BPE Tokenizer - Training Notebook
Author: Claude
Date: November 2025

This notebook trains a Byte Pair Encoding (BPE) tokenizer for Hindi text.
Target: 5000+ tokens vocabulary, 3+ compression ratio
"""

import re
from collections import defaultdict, Counter
import json
from typing import List, Dict, Tuple, Set
import pickle

class HindiBPETokenizer:
    """Custom BPE Tokenizer for Hindi (Devanagari script)"""
    
    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []
        self.byte_encoder = self._create_byte_encoder()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
    def _create_byte_encoder(self) -> Dict[int, str]:
        """Create byte to unicode mapping for handling all bytes"""
        bs = list(range(ord("!"), ord("~")+1)) + \
             list(range(ord("¡"), ord("¬")+1)) + \
             list(range(ord("®"), ord("ÿ")+1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        return dict(zip(bs, [chr(c) for c in cs]))
    
    def _get_stats(self, word_freqs: Dict[Tuple[str, ...], int]) -> Counter:
        """Count frequency of adjacent symbol pairs"""
        pairs = Counter()
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i+1])] += freq
        return pairs
    
    def _merge_pair(self, pair: Tuple[str, str], 
                     word_freqs: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, ...], int]:
        """Merge the most frequent pair in vocabulary"""
        new_word_freqs = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i+1] == pair[1]:
                    new_word.append(replacement)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freqs[tuple(new_word)] = freq
        
        return new_word_freqs
    
    def train(self, texts: List[str], verbose: bool = True):
        """Train BPE tokenizer on Hindi texts"""
        
        # Step 1: Pre-tokenize (split on whitespace and punctuation)
        word_freqs = Counter()
        
        for text in texts:
            # Convert to bytes and then to unicode representation
            text_bytes = text.encode('utf-8')
            text_unicode = ''.join(self.byte_encoder[b] for b in text_bytes)
            
            # Split on whitespace while keeping words
            words = text_unicode.split()
            for word in words:
                # Add end of word marker
                word_with_marker = tuple(word) + ('</w>',)
                word_freqs[word_with_marker] += 1
        
        if verbose:
            print(f"Initial vocabulary size: {len(word_freqs)}")
            print(f"Total words: {sum(word_freqs.values())}")
        
        # Step 2: Build base vocabulary from characters
        self.vocab = {char: idx for idx, char in enumerate(
            sorted(set(char for word in word_freqs for char in word))
        )}
        
        base_vocab_size = len(self.vocab)
        if verbose:
            print(f"Base vocabulary (characters): {base_vocab_size}")
        
        # Step 3: Iteratively merge most frequent pairs
        num_merges = self.vocab_size - base_vocab_size
        
        for i in range(num_merges):
            pairs = self._get_stats(word_freqs)
            
            if not pairs:
                break
            
            best_pair = max(pairs, key=pairs.get)
            word_freqs = self._merge_pair(best_pair, word_freqs)
            
            # Add merged token to vocabulary
            new_token = ''.join(best_pair)
            self.vocab[new_token] = len(self.vocab)
            self.merges.append(best_pair)
            
            if verbose and (i + 1) % 500 == 0:
                print(f"Merge {i+1}/{num_merges}: {best_pair} -> {new_token} "
                      f"(freq: {pairs[best_pair]})")
        
        if verbose:
            print(f"\nFinal vocabulary size: {len(self.vocab)}")
            print(f"Total merges performed: {len(self.merges)}")
    
    def _tokenize_word(self, word: str) -> List[str]:
        """Tokenize a single word using learned merges"""
        word = tuple(word) + ('</w>',)
        
        # Apply merges in order
        for pair in self.merges:
            if len(word) < 2:
                break
            
            i = 0
            new_word = []
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i+1] == pair[1]:
                    new_word.append(''.join(pair))
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
        
        return list(word)
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        # Convert to bytes representation
        text_bytes = text.encode('utf-8')
        text_unicode = ''.join(self.byte_encoder[b] for b in text_bytes)
        
        # Tokenize
        tokens = []
        words = text_unicode.split()
        
        for word in words:
            word_tokens = self._tokenize_word(word)
            for token in word_tokens:
                if token in self.vocab:
                    tokens.append(self.vocab[token])
                else:
                    # Fallback to character level
                    for char in token:
                        if char in self.vocab:
                            tokens.append(self.vocab[char])
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text"""
        # Reverse vocabulary lookup
        id_to_token = {v: k for k, v in self.vocab.items()}
        
        # Get tokens
        tokens = [id_to_token.get(tid, '') for tid in token_ids]
        
        # Join and remove end markers
        text_unicode = ''.join(tokens).replace('</w>', ' ')
        
        # Convert back from byte representation
        try:
            text_bytes = bytes([self.byte_decoder[c] for c in text_unicode if c in self.byte_decoder])
            return text_bytes.decode('utf-8', errors='ignore').strip()
        except:
            return text_unicode.strip()
    
    def save(self, path: str):
        """Save tokenizer to disk"""
        data = {
            'vocab': self.vocab,
            'merges': self.merges,
            'vocab_size': self.vocab_size
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Tokenizer saved to {path}")
    
    def load(self, path: str):
        """Load tokenizer from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.vocab = data['vocab']
        self.merges = data['merges']
        self.vocab_size = data['vocab_size']
        print(f"Tokenizer loaded from {path}")
    
    def calculate_compression_ratio(self, texts: List[str]) -> float:
        """Calculate compression ratio: chars / tokens"""
        total_chars = sum(len(text) for text in texts)
        total_tokens = sum(len(self.encode(text)) for text in texts)
        return total_chars / total_tokens if total_tokens > 0 else 0


# ============================================================================
# TRAINING SCRIPT
# ============================================================================

def create_hindi_dataset() -> List[str]:
    """Create a sample Hindi dataset for training"""
    
    # Sample Hindi texts covering various topics
    hindi_texts = [
        # News/Current Affairs
        "भारत दुनिया का सबसे बड़ा लोकतंत्र है और यहाँ विविध संस्कृतियाँ पाई जाती हैं।",
        "प्रधानमंत्री ने आज एक महत्वपूर्ण बैठक की और नई योजनाओं की घोषणा की।",
        "मुंबई भारत की आर्थिक राजधानी है और यहाँ बॉलीवुड फिल्म उद्योग स्थित है।",
        "भारतीय अंतरिक्ष अनुसंधान संगठन ने चंद्रयान मिशन में सफलता प्राप्त की।",
        
        # Literature/Poetry
        "रात का सन्नाटा, चाँदनी रात, तारों की रोशनी में डूबा हुआ आसमान।",
        "कबीर के दोहे भारतीय साहित्य में अमूल्य रत्न हैं।",
        "हिंदी साहित्य में प्रेमचंद का योगदान अविस्मरणीय है।",
        
        # Science/Technology
        "कृत्रिम बुद्धिमत्ता और मशीन लर्निंग आधुनिक तकनीक के महत्वपूर्ण क्षेत्र हैं।",
        "डिजिटल इंडिया अभियान ने देश में तकनीकी क्रांति लाई है।",
        "भारत में स्मार्टफोन उपयोगकर्ताओं की संख्या तेजी से बढ़ रही है।",
        
        # Daily Life/Conversation
        "आज मौसम बहुत अच्छा है, चलो बाहर घूमने चलते हैं।",
        "मुझे चाय पीना बहुत पसंद है, विशेष रूप से सुबह के समय।",
        "क्या आपने आज का अखबार पढ़ा? वहाँ बहुत दिलचस्प खबरें हैं।",
        "कल मैं बाजार जाऊंगा और सब्जियाँ खरीदूंगा।",
        
        # History/Culture
        "ताजमहल प्यार का प्रतीक है और विश्व के सात अजूबों में से एक है।",
        "भारत की स्वतंत्रता के लिए कई महान नेताओं ने अपना जीवन समर्पित किया।",
        "योग भारत की प्राचीन परंपरा है जो आज पूरे विश्व में लोकप्रिय है।",
        "दिवाली भारत का सबसे बड़ा त्योहार है जो प्रकाश का पर्व है।",
        
        # Education
        "शिक्षा मनुष्य के जीवन में सबसे महत्वपूर्ण है।",
        "छात्रों को नियमित अध्ययन करना चाहिए और अच्छे अंक प्राप्त करने चाहिए।",
        "विज्ञान और गणित आधुनिक शिक्षा के मूलभूत विषय हैं।",
        
        # Sports
        "क्रिकेट भारत में सबसे लोकप्रिय खेल है।",
        "भारतीय खिलाड़ियों ने ओलंपिक में कई पदक जीते हैं।",
        
        # Environment
        "पर्यावरण संरक्षण आज की सबसे बड़ी आवश्यकता है।",
        "वृक्षारोपण से हमें स्वच्छ हवा और बेहतर जीवन मिलता है।",
    ]
    
    # Expand dataset by repeating and adding variations
    expanded_texts = []
    for text in hindi_texts * 20:  # Repeat 20 times
        expanded_texts.append(text)
        # Add variations with common Hindi phrases
        expanded_texts.append(text + " यह बहुत महत्वपूर्ण है।")
        expanded_texts.append("निश्चित रूप से " + text)
    
    return expanded_texts


def main():
    """Main training function"""
    
    print("=" * 70)
    print("HINDI BPE TOKENIZER TRAINING")
    print("=" * 70)
    print()
    
    # Create dataset
    print("Step 1: Creating Hindi dataset...")
    texts = create_hindi_dataset()
    print(f"Dataset size: {len(texts)} texts")
    print(f"Sample text: {texts[0]}")
    print()
    
    # Initialize tokenizer
    print("Step 2: Initializing tokenizer...")
    vocab_size = 5500  # Target: 5000+
    tokenizer = HindiBPETokenizer(vocab_size=vocab_size)
    print(f"Target vocabulary size: {vocab_size}")
    print()
    
    # Train tokenizer
    print("Step 3: Training tokenizer...")
    print("-" * 70)
    tokenizer.train(texts, verbose=True)
    print("-" * 70)
    print()
    
    # Calculate compression ratio
    print("Step 4: Calculating compression ratio...")
    test_texts = texts[:100]  # Use first 100 for testing
    compression_ratio = tokenizer.calculate_compression_ratio(test_texts)
    print(f"Compression Ratio: {compression_ratio:.2f}")
    print()
    
    # Test tokenizer
    print("Step 5: Testing tokenizer...")
    test_sentence = "भारत एक महान देश है और यहाँ विविध संस्कृतियाँ हैं।"
    print(f"Original: {test_sentence}")
    encoded = tokenizer.encode(test_sentence)
    print(f"Encoded: {encoded[:20]}... (showing first 20 tokens)")
    print(f"Number of tokens: {len(encoded)}")
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")
    print()
    
    # Statistics
    print("=" * 70)
    print("FINAL STATISTICS")
    print("=" * 70)
    print(f"✓ Total Token Count: {len(tokenizer.vocab)}")
    print(f"✓ Compression Ratio: {compression_ratio:.2f}")
    print(f"✓ Number of Merges: {len(tokenizer.merges)}")
    print(f"✓ Requirement Check:")
    print(f"  - Tokens > 5000: {'✓ PASS' if len(tokenizer.vocab) > 5000 else '✗ FAIL'}")
    print(f"  - Compression ≥ 3: {'✓ PASS' if compression_ratio >= 3.0 else '✗ FAIL'}")
    print()
    
    # Save tokenizer
    print("Step 6: Saving tokenizer...")
    tokenizer.save('hindi_bpe_tokenizer.pkl')
    print()
    
    # Sample vocabulary
    print("Sample vocabulary entries:")
    sample_vocab = list(tokenizer.vocab.items())[:20]
    for token, idx in sample_vocab:
        print(f"  {idx:4d}: {repr(token)}")
    
    return tokenizer


if __name__ == "__main__":
    tokenizer = main()
