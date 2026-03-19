class ChatMLDataset:
    """
    Returns (tokens, prompt_len) where prompt_len is an int.
    iterate_batches then yields: mx.array(list(zip(offsets, lengths)))
    which becomes lengths[:,0]=prompt_len, lengths[:,1]=total_len
    matching default_loss mask: steps >= prompt_len AND steps <= total_len
    """
    def __init__(self, texts, tokenizer, max_length=512):
        self.data = []
        assistant_header = tokenizer.encode("<|im_start|>assistant\n")
        header_len = len(assistant_header)

        for text in texts:
            tokens = tokenizer.encode(text)
            if len(tokens) > max_length:
                tokens = tokens[:max_length]

            prompt_len = len(tokens)  # fallback
            for i in range(len(tokens) - header_len):
                if tokens[i:i+header_len] == assistant_header:
                    prompt_len = i + header_len
                    break

            if len(tokens) > prompt_len:
                self.data.append((tokens, prompt_len))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def make_dataset(texts, tokenizer, max_length=512):
    return ChatMLDataset(texts, tokenizer, max_length)
