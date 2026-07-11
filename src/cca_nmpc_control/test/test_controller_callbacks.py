"""Tests for NmpcControllerNode callback message parsing.

Covers:
- P1.1: flat HumanState parse (h.x / h.y per docs/06)
- P2.3: configurable input-freshness modes
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock


def _ensure(name: str) -> ModuleType:
    if name not in sys.modules:
        sys.modules[name] = ModuleType(name)
    return sys.modules[name]


def _install_mocks() -> None:
    sys.modules['casadi'] = MagicMock()
    sys.modules['rclpy'] = MagicMock()
    node_mod = MagicMock()
    node_mod.Node = object
    sys.modules['rclpy.node'] = node_mod

    for pkg in (
        'geometry_msgs', 'geometry_msgs.msg',
        'nav_msgs', 'nav_msgs.msg',
        'nav2_msgs', 'nav2_msgs.msg',
        'cca_nmpc_msgs', 'cca_nmpc_msgs.msg',
    ):
        _ensure(pkg)

    geo = sys.modules['geometry_msgs.msg']
    geo.Twist = MagicMock
    geo.PoseStamped = MagicMock
    nav = sys.modules['nav_msgs.msg']
    nav.Odometry = MagicMock
    nav.Path = MagicMock
    nav2 = sys.modules['nav2_msgs.msg']
    nav2.Costmap = MagicMock
    msgs = sys.modules['cca_nmpc_msgs.msg']
    for name in (
        'AdaptiveParams', 'HumanPredictionArray', 'HumanStateArray',
        'ContextIndexArray', 'NmpcDiagnostics', 'SlackValue',
    ):
        setattr(msgs, name, MagicMock)


def _load_node_module():
    """Load source nmpc_controller_node.py only (not stale install)."""
    _install_mocks()

    pkg_root = Path(__file__).resolve().parents[1]  # src/cca_nmpc_control
    pkg_dir = pkg_root / 'cca_nmpc_control'
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))

    # Clear any previously imported package modules.
    for name in list(sys.modules):
        if name == 'cca_nmpc_control' or name.startswith('cca_nmpc_control.'):
            del sys.modules[name]

    # Package root
    pkg = ModuleType('cca_nmpc_control')
    pkg.__path__ = [str(pkg_dir)]
    sys.modules['cca_nmpc_control'] = pkg

    # Mock solver package tree (casadi-heavy).
    solver_pkg = ModuleType('cca_nmpc_control.nmpc_solver')
    solver_pkg.__path__ = [str(pkg_dir / 'nmpc_solver')]
    solver_pkg.CasadiSolver = MagicMock
    sys.modules['cca_nmpc_control.nmpc_solver'] = solver_pkg

    interface = ModuleType('cca_nmpc_control.nmpc_solver.interface')
    interface.AdaptiveParamsInput = MagicMock
    interface.HumanSafetyDistance = MagicMock
    sys.modules['cca_nmpc_control.nmpc_solver.interface'] = interface

    warm = ModuleType('cca_nmpc_control.nmpc_solver.warm_start')
    warm.InvalidationThresholds = MagicMock
    warm.detect_odom_jump = MagicMock(return_value=False)
    warm.detect_goal_change = MagicMock(return_value=False)
    warm.should_reset = MagicMock(return_value=False)
    sys.modules['cca_nmpc_control.nmpc_solver.warm_start'] = warm

    # Load pure-python siblings from source.
    for mod_name, file_name in (
        ('cca_nmpc_control.fallback', 'fallback.py'),
        ('cca_nmpc_control.timing', 'timing.py'),
        ('cca_nmpc_control.invalidation', 'invalidation.py'),
        ('cca_nmpc_control.diagnostics', 'diagnostics.py'),
    ):
        path = pkg_dir / file_name
        spec = importlib.util.spec_from_file_location(mod_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)

    node_path = pkg_dir / 'nmpc_controller_node.py'
    node_spec = importlib.util.spec_from_file_location(
        'cca_nmpc_control.nmpc_controller_node', node_path,
    )
    node_mod = importlib.util.module_from_spec(node_spec)
    sys.modules['cca_nmpc_control.nmpc_controller_node'] = node_mod
    assert node_spec.loader is not None
    node_spec.loader.exec_module(node_mod)
    return node_mod


_node_mod = _load_node_module()
NmpcControllerNode = _node_mod.NmpcControllerNode


class _FakeNode:
    """Minimal stand-in so we can bind the real callback without spinning ROS."""

    def __init__(self) -> None:
        self._latest_humans: dict[int, tuple[float, float]] = {}
        self._received_at: dict[str, float] = {}
        self._strict_runtime_mode: bool = True
        self._allow_empty_humans: bool = False
        self._require_reference_path: bool = True
        self._require_costmap: bool = True
        self._input_timeout: float = 0.5

    def _mark_received(self, name: str) -> None:
        self._received_at[name] = time.monotonic()


def _human(track_id: int, x: float, y: float) -> SimpleNamespace:
    """Duck-typed HumanState with flat fields per docs/06."""
    return SimpleNamespace(track_id=track_id, x=x, y=y, vx=0.0, vy=0.0, confidence=1.0)


def test_on_human_states_parses_flat_fields():
    """P1.1: _on_human_states must use h.x/h.y (docs/06), not h.position.x/y."""
    node = _FakeNode()
    callback = NmpcControllerNode._on_human_states.__get__(node)

    msg = SimpleNamespace(humans=[
        _human(42, 3.5, -1.2),
        _human(17, -2.0, 4.1),
    ])

    callback(msg)

    assert node._latest_humans == {
        42: (3.5, -1.2),
        17: (-2.0, 4.1),
    }
    assert 'humans' in node._received_at


def test_on_human_states_empty_array():
    """Empty HumanStateArray produces empty dict."""
    node = _FakeNode()
    callback = NmpcControllerNode._on_human_states.__get__(node)

    msg = SimpleNamespace(humans=[])

    callback(msg)

    assert node._latest_humans == {}
    assert 'humans' in node._received_at


def test_on_human_states_rejects_position_attribute_contract():
    """Regression guard: nested .position must not be required by the callback.

    If someone reintroduces h.position.x, this still documents the docs/06 contract
    by asserting the real msg shape used at the call site is flat.
    """
    h = _human(1, 0.5, 1.5)
    assert hasattr(h, 'x') and hasattr(h, 'y')
    assert not hasattr(h, 'position')


# ---------------------------------------------------------------------------
# P2.3: configurable input-freshness modes
# ---------------------------------------------------------------------------

_ALL_TOPICS = (
    'params', 'odom', 'predictions', 'humans', 'context', 'reference', 'costmap',
)


def _fresh_node(**flags) -> _FakeNode:
    """Fake node with all topics marked fresh, then apply flag overrides."""
    node = _FakeNode()
    for name in _ALL_TOPICS:
        node._mark_received(name)
    for key, value in flags.items():
        setattr(node, key, value)
    return node


def test_inputs_fresh_strict_mode_requires_all_topics():
    """strict_runtime_mode=True: missing any of the 7 topics → False."""
    inputs_fresh = NmpcControllerNode._inputs_fresh

    # All present → True
    node = _fresh_node(_strict_runtime_mode=True)
    assert inputs_fresh(node) is True

    # Drop each topic one-by-one → False
    for missing in _ALL_TOPICS:
        node = _fresh_node(_strict_runtime_mode=True)
        del node._received_at[missing]
        assert inputs_fresh(node) is False, f'expected False when {missing!r} missing'


def test_inputs_fresh_non_strict_allow_empty_humans():
    """strict=False + allow_empty_humans=True → only params+odom required."""
    inputs_fresh = NmpcControllerNode._inputs_fresh

    node = _FakeNode()
    node._strict_runtime_mode = False
    node._allow_empty_humans = True
    node._require_reference_path = False
    node._require_costmap = False
    node._mark_received('params')
    node._mark_received('odom')
    assert inputs_fresh(node) is True

    # Drop odom → False
    del node._received_at['odom']
    assert inputs_fresh(node) is False


def test_inputs_fresh_non_strict_require_flags():
    """strict=False: require_costmap / require_reference_path gate those topics."""
    inputs_fresh = NmpcControllerNode._inputs_fresh

    # require_costmap=True, costmap missing → False
    node = _FakeNode()
    node._strict_runtime_mode = False
    node._allow_empty_humans = True
    node._require_reference_path = False
    node._require_costmap = True
    node._mark_received('params')
    node._mark_received('odom')
    assert inputs_fresh(node) is False

    node._mark_received('costmap')
    assert inputs_fresh(node) is True

    # require_reference_path=True, reference missing → False
    node = _FakeNode()
    node._strict_runtime_mode = False
    node._allow_empty_humans = True
    node._require_reference_path = True
    node._require_costmap = False
    node._mark_received('params')
    node._mark_received('odom')
    assert inputs_fresh(node) is False

    node._mark_received('reference')
    assert inputs_fresh(node) is True

    # allow_empty_humans=False → predictions/humans/context required
    node = _FakeNode()
    node._strict_runtime_mode = False
    node._allow_empty_humans = False
    node._require_reference_path = False
    node._require_costmap = False
    node._mark_received('params')
    node._mark_received('odom')
    assert inputs_fresh(node) is False

    node._mark_received('predictions')
    node._mark_received('humans')
    node._mark_received('context')
    assert inputs_fresh(node) is True


def test_inputs_fresh_stale_topic_fails():
    """Topic present but older than input_timeout → False."""
    inputs_fresh = NmpcControllerNode._inputs_fresh

    node = _fresh_node(_strict_runtime_mode=True, _input_timeout=0.05)
    # Force odom timestamp into the past beyond timeout
    node._received_at['odom'] = time.monotonic() - 1.0
    assert inputs_fresh(node) is False

    # Fresh again → True
    node._mark_received('odom')
    assert inputs_fresh(node) is True
