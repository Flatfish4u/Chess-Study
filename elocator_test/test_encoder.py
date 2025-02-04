from encoder import fen_encoder

def test_encoder():
    # Test starting position
    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    encoded = fen_encoder(start_fen)
    
    print("Encoder Test:")
    print(f"Input FEN: {start_fen}")
    print(f"Output vector shape: {encoded.shape}")
    print(f"Expected shape: 780 (8x8x12 + 8 +