from .physics_engine import DropletPhysics
from .perspective import PerspectiveCorrector

try:
    from .ai_engine import AIContactAngleAnalyzer
except ImportError:
    # If mobile_sam or torch is missing
    AIContactAngleAnalyzer = None
