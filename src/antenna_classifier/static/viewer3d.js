/**
 * Three.js 3D antenna viewer for NEC geometry.
 * ES module — loaded via importmap for Three.js resolution.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const COLORS = {
  wire:      0x58a6ff,
  excited:   0xf85149,
  loaded:    0xd29922,
  ground:    0x3fb950,
  grid:      0x30363d,
  bg:        0x0d1117,
  feedPoint: 0xff4444,
};

let _scene, _camera, _renderer, _controls, _animId, _resizeObs;

export function dispose() {
  if (_animId) cancelAnimationFrame(_animId);
  if (_controls) _controls.dispose();
  if (_resizeObs) _resizeObs.disconnect();
  if (_renderer) {
    _renderer.dispose();
    _renderer.domElement.remove();
  }
  _scene = _camera = _renderer = _controls = _animId = _resizeObs = null;
}

export function init(containerId, geometry) {
  dispose();

  const container = document.getElementById(containerId);
  if (!container) return;

  const w = container.clientWidth || 600;
  const h = container.clientHeight || 500;

  // Scene
  _scene = new THREE.Scene();
  _scene.background = new THREE.Color(COLORS.bg);

  // Camera
  _camera = new THREE.PerspectiveCamera(50, w / h, 0.001, 10000);

  // Renderer
  _renderer = new THREE.WebGLRenderer({ antialias: true });
  _renderer.setSize(w, h);
  _renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(_renderer.domElement);

  // Controls
  _controls = new OrbitControls(_camera, _renderer.domElement);
  _controls.enableDamping = true;
  _controls.dampingFactor = 0.1;

  // Lights
  _scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
  dirLight.position.set(5, 10, 7);
  _scene.add(dirLight);

  // Parse bounds (NEC: X right, Y forward, Z up)
  const b = geometry.bounds;
  // Center in Three.js coords (swap Y↔Z)
  const center = new THREE.Vector3(
    (b.min[0] + b.max[0]) / 2,
    (b.min[2] + b.max[2]) / 2,
    -(b.min[1] + b.max[1]) / 2,
  );
  const sceneSize = Math.max(
    b.max[0] - b.min[0],
    b.max[1] - b.min[1],
    b.max[2] - b.min[2],
    0.1,
  );

  const wireVisRadius = sceneSize * 0.004;

  // Add wires
  for (const wire of (geometry.wires || [])) {
    const color = wire.is_excited ? COLORS.excited
               : wire.is_loaded  ? COLORS.loaded
               : COLORS.wire;
    addWire(wire, color, wireVisRadius);

    // Feed-point markers
    if (wire.is_excited && wire.points.length >= 2) {
      for (const seg of (wire.excited_segments || [])) {
        const pos = segMidpoint(wire.points, wire.segments, seg);
        addMarker(pos, COLORS.feedPoint, sceneSize * 0.012);
      }
    }
  }

  // Ground plane
  if (geometry.ground_type && geometry.ground_type !== 'none' && geometry.ground_type !== 'free_space') {
    const gs = sceneSize * 3;
    const gnd = new THREE.Mesh(
      new THREE.PlaneGeometry(gs, gs),
      new THREE.MeshPhongMaterial({
        color: COLORS.ground, transparent: true, opacity: 0.12, side: THREE.DoubleSide,
      }),
    );
    gnd.rotation.x = -Math.PI / 2;
    gnd.position.set(center.x, 0, center.z);
    _scene.add(gnd);
  }

  // Grid
  const grid = new THREE.GridHelper(sceneSize * 2, 10, COLORS.grid, COLORS.grid);
  grid.position.set(center.x, Math.min(0, b.min[2]), center.z);
  _scene.add(grid);

  // Axes (Red=X, Green=Y/Z-up, Blue=Z/Y-fwd)
  _scene.add(new THREE.AxesHelper(sceneSize * 0.25));

  // Camera position
  const dist = sceneSize * 2.2;
  _camera.position.set(
    center.x + dist * 0.7,
    center.y + dist * 0.9,
    center.z + dist * 0.7,
  );
  _controls.target.copy(center);
  _controls.update();

  // Render loop
  (function animate() {
    _animId = requestAnimationFrame(animate);
    _controls.update();
    _renderer.render(_scene, _camera);
  })();

  // Resize
  _resizeObs = new ResizeObserver(() => {
    if (!_renderer) return;
    const nw = container.clientWidth || 600;
    const nh = container.clientHeight || 500;
    _camera.aspect = nw / nh;
    _camera.updateProjectionMatrix();
    _renderer.setSize(nw, nh);
  });
  _resizeObs.observe(container);
}

/* ---- helpers ---- */

/** NEC (X, Y, Z-up) → Three.js (X, Y-up, -Z) */
function nec(pt) {
  return new THREE.Vector3(pt[0], pt[2], -pt[1]);
}

function addWire(wire, color, visRadius) {
  const pts = wire.points;
  if (pts.length < 2) return;

  const mat = new THREE.MeshPhongMaterial({
    color, emissive: color, emissiveIntensity: 0.15,
  });

  // Cylinder per segment + sphere at each joint
  for (let i = 0; i < pts.length - 1; i++) {
    const p1 = nec(pts[i]);
    const p2 = nec(pts[i + 1]);
    const dir = new THREE.Vector3().subVectors(p2, p1);
    const len = dir.length();
    if (len < 1e-10) continue;

    const cyl = new THREE.Mesh(
      new THREE.CylinderGeometry(visRadius, visRadius, len, 6),
      mat,
    );
    cyl.position.lerpVectors(p1, p2, 0.5);
    cyl.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
    _scene.add(cyl);
  }

  // Joint spheres
  for (const pt of pts) {
    const sph = new THREE.Mesh(new THREE.SphereGeometry(visRadius, 6, 6), mat);
    sph.position.copy(nec(pt));
    _scene.add(sph);
  }
}

function addMarker(position, color, radius) {
  const mat = new THREE.MeshPhongMaterial({
    color, emissive: color, emissiveIntensity: 0.4,
  });
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(radius, 12, 12), mat);
  mesh.position.copy(nec(position));
  _scene.add(mesh);
}

function segMidpoint(points, totalSegments, segIndex) {
  const t = Math.max(0, Math.min(1, (segIndex - 0.5) / Math.max(totalSegments, 1)));

  if (points.length === 2) {
    return [
      points[0][0] + (points[1][0] - points[0][0]) * t,
      points[0][1] + (points[1][1] - points[0][1]) * t,
      points[0][2] + (points[1][2] - points[0][2]) * t,
    ];
  }

  const idx = t * (points.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.min(lo + 1, points.length - 1);
  const frac = idx - lo;
  return [
    points[lo][0] + (points[hi][0] - points[lo][0]) * frac,
    points[lo][1] + (points[hi][1] - points[lo][1]) * frac,
    points[lo][2] + (points[hi][2] - points[lo][2]) * frac,
  ];
}
