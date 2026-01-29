# python/utils/mock_dependencies.py
import sys
import builtins

def apply_mocks():
    """Apply mocks for six and pyaudio to avoid dependency issues."""

    # Mock six.moves.queue
    class MockSixMovesQueue:
        Queue = None

    class MockSixMoves:
        queue = MockSixMovesQueue()

    class MockSix:
        moves = MockSixMoves()

    sys.modules['six'] = MockSix()
    sys.modules['six.moves'] = MockSixMoves()
    sys.modules['six.moves.queue'] = MockSixMovesQueue()

    # Mock pyaudio
    class MockPyAudio:
        pass

    sys.modules['pyaudio'] = MockPyAudio()
    builtins.pyaudio = MockPyAudio()