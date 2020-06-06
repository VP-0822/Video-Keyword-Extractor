import torch.nn as nn
from embedding import VocabularyEmbedder, FeatureEmbedder
from positionalEncoder import PositionalEncoder
from encoder import Encoder
from decoder import Decoder
from generator import Generator

class MultiModalTransformer(nn.Module):
    """
        MultiModalTransformer transformes multi-modal (subtitle, audio and video) inputs to caption outputs.
    """
    
    def __init__(self, caption_vocab_size, subtitle_vocab_size,
                 audio_feature_dimension, video_feature_dimension,
                 audio_model_dimension, video_model_dimension, subtitle_model_dimension,
                 audio_feedforward_dimension, video_feedforward_dimension, subtitle_feedforward_dimension,
                 audio_number_of_layers, video_number_of_layers, subtitle_number_of_layers,
                 dropout_percentage, number_of_heads, use_linear_embedder):
        """
            Creates necessary layers for complete encoder-decoder-generator transformer archiecture layers.
            Applying uniform xavier initializer for the parameters
            Args:
                caption_vocab_size: total caption vocabulary size
                subtitle_vocab_size: total subtitle vocabulary size
                audio_feature_dimension: Input feature dimension for audio modality
                video_feature_dimension: Input feature dimension for video modality
                audio_model_dimension: Audio model dimensions for encoder-decoder framework
                video_model_dimension: Video model dimensions for encoder-decoder framework
                subtitle_model_dimension: Subtitle model dimensions for encoder-decoder framework
                audio_feedforward_dimension: Audio position-wise feed forward units
                video_feedforward_dimension: Video position-wise feed forward units
                subtitle_feedforward_dimension: Subtitle position-wise feed forward units
                audio_number_of_layers: Number of layers for encoder and decoder for audio modality
                video_number_of_layers: Number of layers for encoder and decoder for video modality
                subtitle_number_of_layers: Number of layers for encoder and decoder for subtitle modality
                dropout_percentage: dropout percentage at various modules.
                number_of_heads: Number of heads in multi-headed attention
                use_linear_embedder: Use linear dense layer at inputs for audio and video modalities
        """
        super(MultiModalTransformer, self).__init__()
        self.src_emb_subs = VocabularyEmbedder(subtitle_vocab_size, subtitle_model_dimension)
        if use_linear_embedder:
            self.src_emb_audio = FeatureEmbedder(audio_feature_dimension, audio_model_dimension)
            self.src_emb_video = FeatureEmbedder(video_feature_dimension, video_model_dimension)
        else:
            assert video_feature_dimension == video_model_dimension and audio_feature_dimension == audio_model_dimension
            self.src_emb_audio = Identity()
            self.src_emb_video = Identity()
        
        self.trg_emb_subs  = VocabularyEmbedder(caption_vocab_size, subtitle_model_dimension)
        self.trg_emb_audio = VocabularyEmbedder(caption_vocab_size, audio_model_dimension)
        self.trg_emb_video = VocabularyEmbedder(caption_vocab_size, video_model_dimension)
        self.pos_emb_subs  = PositionalEncoder(subtitle_model_dimension, dropout_percentage)
        self.pos_emb_audio = PositionalEncoder(audio_model_dimension, dropout_percentage)
        self.pos_emb_video = PositionalEncoder(video_model_dimension, dropout_percentage)
        self.encoder_subs =  Encoder(subtitle_model_dimension,  dropout_percentage, number_of_heads, subtitle_feedforward_dimension,  subtitle_number_of_layers)
        self.encoder_audio = Encoder(audio_model_dimension, dropout_percentage, number_of_heads, audio_feedforward_dimension, audio_number_of_layers)
        self.encoder_video = Encoder(video_model_dimension, dropout_percentage, number_of_heads, video_feedforward_dimension, video_number_of_layers)
        self.decoder_subs =  Decoder(subtitle_model_dimension,  dropout_percentage, number_of_heads, subtitle_feedforward_dimension,  subtitle_number_of_layers)
        self.decoder_audio = Decoder(audio_model_dimension, dropout_percentage, number_of_heads, audio_feedforward_dimension, audio_number_of_layers)
        self.decoder_video = Decoder(video_model_dimension, dropout_percentage, number_of_heads, video_feedforward_dimension, video_number_of_layers)
        
        # late fusion
        self.generator = Generator(
            subtitle_model_dimension, audio_model_dimension, video_model_dimension, caption_vocab_size, dropout_percentage
        )
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    # src_subs (B, Ss2, d_feat_subs), src_audio (B, Ss, d_feat_audio) src_video (B, Ss, d_feat_video) 
    # trg (B, St) src_mask (B, 1, Ss) src_sub_mask (B, 1, Ssubs) trg_mask (B, St, St)
    def forward(self, src, trg, mask):
        """
            Following steps are performed,
            1. Apply VocabularyEmbedding for subtitles and FeatureEmbedding for audio and video inputs
            2. Create target embedding for subtitle, audio and video modalities using VocabularyEmbedding
            3. Create positional encodings for subtitles, audio and video for both source and target embeddings
            4. Apply encoder layer to subtitle, audio and video modalities and store encoder memories
            5. Decoder subtitle, audio and video encoder memories using target positional embeddings
            6. Pass decoder outputs for all three modalities to generator module.
            Args:
                src: it has subtitle (batch_size, subtitle_sequence_length, subtitle_model_dimension), audio (batch_size, sequence_length, audio_model_dimension) 
                    and video (batch_size, sequence_length, video_model_dimension) sources
                trg: target caption word by word of shape (batch_size, target_sequence_length)
                mask: it has masks for audio and video of shape (batch_size, 1, sequence_length), subtitle mask of shape
                    (batch_size, 1, subtitle_sequence_length) and target mask of shape (batch_size, target_sequence_length, target_sequence_length)
            Returns:
                Output of transformer of shape (batch_size, target_sequence_length, caption_vocab_size)
        """
        src_video, src_audio, src_subs = src
        src_mask, trg_mask, src_subs_mask = mask

        # embed
        src_subs = self.src_emb_subs(src_subs)
        src_audio = self.src_emb_audio(src_audio)
        src_video = self.src_emb_video(src_video)
        
        trg_subs = self.trg_emb_subs(trg)
        trg_audio = self.trg_emb_audio(trg)
        trg_video = self.trg_emb_video(trg)
        
        src_subs = self.pos_emb_subs(src_subs)
        src_audio = self.pos_emb_audio(src_audio)
        src_video = self.pos_emb_video(src_video)
        
        trg_subs = self.pos_emb_subs(trg_subs)
        trg_audio = self.pos_emb_audio(trg_audio)
        trg_video = self.pos_emb_video(trg_video)
        
        # encode and decode
        memory_subs = self.encoder_subs(src_subs, src_subs_mask)
        memory_audio = self.encoder_audio(src_audio, src_mask)
        memory_video = self.encoder_video(src_video, src_mask)
        
        out_subs = self.decoder_subs(trg_subs, memory_subs, src_subs_mask, trg_mask)
        out_audio = self.decoder_audio(trg_audio, memory_audio, src_mask, trg_mask)
        out_video = self.decoder_video(trg_video, memory_video, src_mask, trg_mask)
        
        # generate
        out = self.generator(out_subs, out_audio, out_video)
        
        return out # (B, St, voc_size)

class Identity(nn.Module):
    """
        creates identity module which returns inputs as is outputs
    """
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        """
            returns inputs as outputs
            Args:
                x: inputs
            Returns:
                inputs as is outputs
        """
        return x