# ============================================================
# chatbot_module.py  (LOCAL CHATBOT USING DialoGPT)
#
# TEAM MEMBER (RESPONSIBLE):
# ➤ Mohammad Adil – Chatbot Integration Specialist
#
# PURPOSE:
# Ye module ek local conversational chatbot banata hai jo
# DialoGPT-medium model use karta hai. Yeh user ke messages
# ko encode karta hai, history maintain karta hai aur phir
# response generate karta hai.
#
# FLOW:
# app.py → bot = LocalChatbot()
# app.py → reply = bot.ask("user message")
# ============================================================

from transformers import AutoModelForCausalLM, AutoTokenizer  # Model + Tokenizer
import torch  # Tensor operations and model inference


class LocalChatbot:
    """
    LocalChatbot class:
    - Model load karta hai
    - Tokenizer load karta hai
    - Conversation history maintain karta hai
    - Response generate karta hai
    """

    def __init__(self):
        """
        -- YEH METHOD sirf EK BAAR run hota hai --
        Jab app.py me LocalChatbot() create hota hai.

        Iske andar:
        1. Model path set hota hai
        2. Tokenizer load hota hai
        3. Model load hota hai
        4. Chat history ko None se initialize kiya jata hai
        """

        # Local model ka path (DialoGPT-medium)
        model_path = "models/DialoGPT-medium"

        # Tokenizer load (text → tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Pre-trained model load (language generation)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)

        # Chat history (pooray conversation ka token history rakhta hai)
        self.chat_history_ids = None


    def ask(self, user_input):
        """
        Yeh method tab call hota hai jab user koi msg bhejta hai.

        STEPS:
        --------------------------------------------------------
        1. User ke message ko token IDs me convert karo
        2. Agar pehle ka conversation hai to append karo
        3. Model.generate() se response tokens banao
        4. Sirf naya bot reply extract karo
        5. Response decode karo text me
        6. Return final reply
        --------------------------------------------------------
        """

        # STEP 1 → USER INPUT ko encode (EOS = end-of-sentence token)
        new_input_ids = self.tokenizer.encode(
            user_input + self.tokenizer.eos_token,  # EOS token append
            return_tensors='pt'
        )

        # STEP 2 → History exist karti hai?
        if self.chat_history_ids is not None:
            # Agar haan → purani history + naya input concatenate
            bot_input_ids = torch.cat([self.chat_history_ids, new_input_ids], dim=-1)
        else:
            # Agar pehli baar message aa raha hai
            bot_input_ids = new_input_ids

        # STEP 3 → Model ka response generate karo
        # model.generate():
        # - Poora conversation padhta hai
        # - Next possible tokens predict karta hai
        self.chat_history_ids = self.model.generate(
            bot_input_ids,
            max_length=1000,  # Maximum combined length
            pad_token_id=self.tokenizer.eos_token_id
        )

        # STEP 4 → Sirf BOT ka naya response extract karo
        # Pura history + user message ke baad jo tokens aaye wo bot ka reply hai
        response = self.tokenizer.decode(
            self.chat_history_ids[:, bot_input_ids.shape[-1]:][0],
            skip_special_tokens=True
        )

        # STEP 5 → Final reply return
        return response
