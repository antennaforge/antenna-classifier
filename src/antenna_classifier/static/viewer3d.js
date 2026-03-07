/**
 * Three.js 3D antenna viewer for NEC geometry.
 * Engineering-grade visualization with:
 *   - Ground plane + height annotation
 *   - Feed point / transmission line markers with tooltips
 *   - Load markers with tooltips
 *   - Labeled XYZ axes with compass rose
 *   - Element dimension/spacing readout
 *   - Info panel overlay
 *
 * ES module — loaded via importmap for Three.js resolution.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

const COLORS = {
  wire:       0x58a6ff,
  excited:    0xf85149,
  loaded:     0xd29922,
  ground:     0x3fb950,
  grid:       0x30363d,
  bg:         0x0d1117,
  feedPoint:  0xff4444,
  tlLine:     0xd29922,
  loadMarker: 0xff9800,
  axisX:      0xff4444,
  axisY:      0x44ff44,
  axisZ:      0x4488ff,
  dimLine:    0x8b949e,
};

let _scene, _camera, _renderer, _labelRenderer, _controls, _animId, _resizeObs;
let _tooltip, _container, _infoPanel;
let _geometry, _wireVisRadius;
let _currentMeshes = [];  // meshes added by showCurrents()
let _originalWireMeshes = [];  // { tag, meshes[] } for hide/show

export function dispose() {
  if (_animId) cancelAnimationFrame(_animId);
  if (_controls) _controls.dispose();
  if (_resizeObs) _resizeObs.disconnect();
  if (_renderer) { _renderer.dispose(); _renderer.domElement.remove(); }
  if (_labelRenderer) _labelRenderer.domElement.remove();
  if (_tooltip) _tooltip.remove();
  if (_infoPanel) _infoPanel.remove();
  _scene = _camera = _renderer = _labelRenderer = _controls = _animId = _resizeObs = null;
  _tooltip = _container = _infoPanel = null;
  _geometry = null; _wireVisRadius = 0;
  _currentMeshes = []; _originalWireMeshes = [];
}

/* ================================================================
   MAIN INIT
   ================================================================ */
export function init(containerId, geometry) {
  dispose();

  _container = document.getElementById(containerId);
  if (!_container) return;

  const w = _container.clientWidth || 600;
  const h = _container.clientHeight || 500;

  // Scene
  _scene = new THREE.Scene();
  _scene.background = new THREE.Color(COLORS.bg);

  // Camera
  _camera = new THREE.PerspectiveCamera(50, w / h, 0.001, 50000);

  // WebGL renderer
  _renderer = new THREE.WebGLRenderer({ antialias: true });
  _renderer.setSize(w, h);
  _renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  _container.appendChild(_renderer.domElement);

  // CSS2D renderer (for labels)
  _labelRenderer = new CSS2DRenderer();
  _labelRenderer.setSize(w, h);
  _labelRenderer.domElement.style.position = 'absolute';
  _labelRenderer.domElement.style.top = '0';
  _labelRenderer.domElement.style.left = '0';
  _labelRenderer.domElement.style.pointerEvents = 'none';
  _container.style.position = 'relative';
  _container.appendChild(_labelRenderer.domElement);

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
  const center = new THREE.Vector3(
    (b.min[0] + b.max[0]) / 2,
    (b.min[2] + b.max[2]) / 2, // Z→Y
    -(b.min[1] + b.max[1]) / 2,  // Y→-Z
  );
  const sceneSize = Math.max(
    b.max[0] - b.min[0],
    b.max[1] - b.min[1],
    b.max[2] - b.min[2],
    0.1,
  );
  const wireVisRadius = sceneSize * 0.004;
  _wireVisRadius = wireVisRadius;
  _geometry = geometry;

  // Tooltip element (HTML overlay)
  _tooltip = document.createElement('div');
  Object.assign(_tooltip.style, {
    position: 'absolute', display: 'none', pointerEvents: 'none',
    background: 'rgba(22,26,46,0.95)', color: '#e8ecf4',
    border: '1px solid #2a3050', borderRadius: '6px', padding: '8px 12px',
    fontSize: '12px', fontFamily: 'monospace', lineHeight: '1.5',
    zIndex: '100', maxWidth: '300px', whiteSpace: 'pre-line',
  });
  _container.appendChild(_tooltip);

  // Raycaster for hover tooltips
  const raycaster = new THREE.Raycaster();
  const mouse = new THREE.Vector2();
  const hoverables = []; // { mesh, text }

  /* ---- 1. Wires ---- */
  _originalWireMeshes = [];
  for (const wire of (geometry.wires || [])) {
    const color = wire.is_excited ? COLORS.excited
               : wire.is_loaded  ? COLORS.loaded
               : COLORS.wire;
    const meshes = addWire(wire, color, wireVisRadius);
    _originalWireMeshes.push({ tag: wire.tag, meshes });
  }

  /* ---- 2. Ground plane + height annotation ---- */
  const gt = geometry.ground_type || 'none';
  const minZ = b.min[2];
  const maxZ = b.max[2];
  const hasGround = gt !== 'none' && gt !== 'free_space';
  const groundLabel = hasGround ? (gt === 'perfect' ? 'Perfect Ground' : 'Real Ground')
                                : 'Free Space (no ground)';

  // Always show ground reference plane at z=0
  {
    const gs = Math.max(sceneSize * 3, 2);
    const gnd = new THREE.Mesh(
      new THREE.PlaneGeometry(gs, gs),
      new THREE.MeshPhongMaterial({
        color: hasGround ? COLORS.ground : 0x30363d,
        transparent: true,
        opacity: hasGround ? 0.15 : 0.06,
        side: THREE.DoubleSide,
      }),
    );
    gnd.rotation.x = -Math.PI / 2;
    gnd.position.set(center.x, 0, center.z);
    _scene.add(gnd);

    // "z = 0" label on ground
    const gndLbl = makeLabel('z = 0  ' + groundLabel, '#3fb950', 11);
    gndLbl.position.set(center.x + sceneSize * 0.6, 0.01, center.z + sceneSize * 0.6);
    _scene.add(gndLbl);
  }

  // Height annotation (ruler line from z=0 to element height)
  if (minZ > 0.01 || maxZ > 0.01) {
    const avgZ = (minZ + maxZ) / 2;
    const rulerX = b.max[0] + sceneSize * 0.15;
    const rulerZ = -(b.min[1] + b.max[1]) / 2;

    // Vertical line
    const rlMat = new THREE.LineBasicMaterial({ color: 0x8b949e, linewidth: 1 });
    const rlGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(rulerX, 0, rulerZ),
      new THREE.Vector3(rulerX, avgZ, rulerZ),
    ]);
    _scene.add(new THREE.Line(rlGeo, rlMat));

    // Tick marks
    const tickW = sceneSize * 0.03;
    for (const y of [0, avgZ]) {
      const tGeo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(rulerX - tickW, y, rulerZ),
        new THREE.Vector3(rulerX + tickW, y, rulerZ),
      ]);
      _scene.add(new THREE.Line(tGeo, rlMat));
    }

    // Height label
    const hLabel = makeLabel(avgZ.toFixed(1) + ' m', '#e8ecf4', 11);
    hLabel.position.set(rulerX + sceneSize * 0.06, avgZ / 2, rulerZ);
    _scene.add(hLabel);
  }

  /* ---- 3. Feed-point and TL markers with tooltips ---- */
  const wireMap = {};
  for (const wire of (geometry.wires || [])) {
    wireMap[wire.tag] = wire;
  }

  // Excitation markers
  for (const ex of (geometry.excitations || [])) {
    const wire = wireMap[ex.tag];
    if (!wire) continue;
    const pos = segMidpoint(wire.points, wire.segments, ex.segment);
    const markerSize = sceneSize * 0.015;
    const mesh = addMarker(pos, COLORS.feedPoint, markerSize, 'diamond');
    const vMag = Math.sqrt(ex.v_real * ex.v_real + ex.v_imag * ex.v_imag).toFixed(3);
    const vPhase = (Math.atan2(ex.v_imag, ex.v_real) * 180 / Math.PI).toFixed(1);
    hoverables.push({
      mesh,
      text: 'FEED: EX wire ' + ex.tag + ' seg ' + ex.segment +
            '\nV = ' + vMag + ' \u2220' + vPhase + '\u00b0' +
            '\n(' + ex.v_real + ' + j' + ex.v_imag + ')',
    });
  }

  // Transmission line markers + dashed connector
  for (const tl of (geometry.transmission_lines || [])) {
    const w1 = wireMap[tl.tag1], w2 = wireMap[tl.tag2];
    if (!w1 || !w2) continue;
    const pos1 = segMidpoint(w1.points, w1.segments, tl.seg1);
    const pos2 = segMidpoint(w2.points, w2.segments, tl.seg2);

    // Markers at both ends
    const m1 = addMarker(pos1, COLORS.tlLine, sceneSize * 0.012, 'ring');
    const m2 = addMarker(pos2, COLORS.tlLine, sceneSize * 0.012, 'ring');

    const tlText = 'TL: wire ' + tl.tag1 + ':' + tl.seg1 +
                   ' \u2192 wire ' + tl.tag2 + ':' + tl.seg2 +
                   '\nZ\u2080 = ' + tl.z0 + ' \u03a9' +
                   (tl.length ? '\nLength = ' + tl.length + ' m' : ' (electrical)');
    hoverables.push({ mesh: m1, text: tlText });
    hoverables.push({ mesh: m2, text: tlText });

    // Dashed line between TL endpoints
    const dashMat = new THREE.LineDashedMaterial({
      color: COLORS.tlLine, dashSize: sceneSize * 0.02,
      gapSize: sceneSize * 0.01, linewidth: 1,
    });
    const dashGeo = new THREE.BufferGeometry().setFromPoints([nec(pos1), nec(pos2)]);
    const dashLine = new THREE.Line(dashGeo, dashMat);
    dashLine.computeLineDistances();
    _scene.add(dashLine);
  }

  /* ---- 4. Load markers ---- */
  for (const ld of (geometry.loads || [])) {
    if (ld.ld_type === 5) continue; // wire conductivity — not a point load
    const wire = wireMap[ld.tag];
    if (!wire || !ld.seg_start) continue;
    const seg = Math.floor((ld.seg_start + ld.seg_end) / 2) || ld.seg_start;
    const pos = segMidpoint(wire.points, wire.segments, seg);
    const mesh = addMarker(pos, COLORS.loadMarker, sceneSize * 0.01, 'cube');

    let desc = 'LOAD: ' + ld.type_name + '\nWire ' + ld.tag + ' seg ' + ld.seg_start;
    if (ld.seg_end !== ld.seg_start) desc += '-' + ld.seg_end;
    if (ld.ld_type <= 1) {
      const parts = [];
      if (ld.zlr) parts.push('R=' + fmtEng(ld.zlr) + '\u03a9');
      if (ld.zli) parts.push('L=' + fmtEng(ld.zli) + 'H');
      if (ld.zlc) parts.push('C=' + fmtEng(ld.zlc) + 'F');
      if (parts.length) desc += '\n' + parts.join('  ');
    }
    hoverables.push({ mesh, text: desc });
  }

  /* ---- 5. XYZ axes with compass labels ---- */
  {
    const axLen = sceneSize * 0.4;
    const origin = new THREE.Vector3(
      b.min[0] - sceneSize * 0.15,
      0,
      -(b.min[1]) + sceneSize * 0.15,
    );

    // X axis (red)
    addAxisArrow(origin, new THREE.Vector3(1, 0, 0), axLen, COLORS.axisX);
    const xlbl = makeLabel('X', '#ff4444', 12, true);
    xlbl.position.set(origin.x + axLen * 1.12, origin.y, origin.z);
    _scene.add(xlbl);

    // Y axis (green) — NEC Z-up → Three.js Y-up
    addAxisArrow(origin, new THREE.Vector3(0, 1, 0), axLen, COLORS.axisY);
    const ylbl = makeLabel('Z (up)', '#44ff44', 12, true);
    ylbl.position.set(origin.x, origin.y + axLen * 1.12, origin.z);
    _scene.add(ylbl);

    // Z axis (blue) — NEC Y-fwd → Three.js -Z
    addAxisArrow(origin, new THREE.Vector3(0, 0, -1), axLen, COLORS.axisZ);
    const zlbl = makeLabel('Y (fwd)', '#4488ff', 12, true);
    zlbl.position.set(origin.x, origin.y, origin.z - axLen * 1.12);
    _scene.add(zlbl);

    // Compass cardinal directions on ground plane
    const compassR = sceneSize * 0.9;
    const compassPairs = [
      ['0\u00b0', 1, 0], ['90\u00b0', 0, -1], ['180\u00b0', -1, 0], ['270\u00b0', 0, 1],
    ];
    for (const [label, dx, dz] of compassPairs) {
      const lbl = makeLabel(label, '#8b949e', 10);
      lbl.position.set(center.x + dx * compassR, 0.01, center.z + dz * compassR);
      _scene.add(lbl);
    }
  }

  /* ---- 6. Dimension annotations ---- */
  const dimY = (minZ > 0.01) ? -sceneSize * 0.08 : minZ - sceneSize * 0.08;
  const wireInfos = geometry.wire_dimensions || [];
  const spacings = geometry.spacings || [];

  // Wire lengths — annotate each wire (hover tooltip on invisible mesh)
  for (const wi of wireInfos) {
    const wire = wireMap[wi.tag];
    if (!wire || wire.points.length < 2) continue;
    const p1 = wire.points[0], p2 = wire.points[wire.points.length - 1];
    const dimHover = addDimensionLine(p1, p2, wi.length.toFixed(2) + ' m', dimY);
    if (dimHover) hoverables.push(dimHover);
  }

  // Spacings — annotate distance between wire midpoints (hover tooltip)
  for (const sp of spacings) {
    const w1 = wireInfos.find(function(wi) { return wi.tag === sp.wire_a; });
    const w2 = wireInfos.find(function(wi) { return wi.tag === sp.wire_b; });
    if (!w1 || !w2) continue;
    const spHover = addSpacingLine(w1.midpoint, w2.midpoint, sp.distance.toFixed(2) + ' m');
    if (spHover) hoverables.push(spHover);
  }

  /* ---- 7. Grid ---- */
  const gridY = hasGround ? 0 : Math.min(0, minZ);
  const grid = new THREE.GridHelper(sceneSize * 2, 10, COLORS.grid, COLORS.grid);
  grid.position.set(center.x, gridY, center.z);
  _scene.add(grid);

  /* ---- 8. Info panel overlay ---- */
  buildInfoPanel(geometry, wireInfos, spacings);

  /* ---- Camera ---- */
  const dist = sceneSize * 2.2;
  _camera.position.set(
    center.x + dist * 0.7,
    center.y + dist * 0.9,
    center.z + dist * 0.7,
  );
  _controls.target.copy(center);
  _controls.update();

  /* ---- Render loop ---- */
  (function animate() {
    _animId = requestAnimationFrame(animate);
    _controls.update();
    _renderer.render(_scene, _camera);
    _labelRenderer.render(_scene, _camera);
  })();

  /* ---- Hover tooltip handling ---- */
  _renderer.domElement.addEventListener('mousemove', function(e) {
    const rect = _renderer.domElement.getBoundingClientRect();
    mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, _camera);
    let hit = null;
    for (const h of hoverables) {
      const intersects = raycaster.intersectObject(h.mesh);
      if (intersects.length > 0) { hit = h; break; }
    }
    if (hit) {
      _tooltip.textContent = hit.text;
      _tooltip.style.display = 'block';
      _tooltip.style.left = (e.clientX - rect.left + 14) + 'px';
      _tooltip.style.top = (e.clientY - rect.top + 14) + 'px';
    } else {
      _tooltip.style.display = 'none';
    }
  });

  /* ---- Resize ---- */
  _resizeObs = new ResizeObserver(function() {
    if (!_renderer) return;
    const nw = _container.clientWidth || 600;
    const nh = _container.clientHeight || 500;
    _camera.aspect = nw / nh;
    _camera.updateProjectionMatrix();
    _renderer.setSize(nw, nh);
    _labelRenderer.setSize(nw, nh);
  });
  _resizeObs.observe(_container);
}


/* ================================================================
   HELPERS
   ================================================================ */

/** NEC (X, Y, Z-up) → Three.js (X, Y-up, -Z) */
function nec(pt) {
  return new THREE.Vector3(pt[0], pt[2], -pt[1]);
}

/** Format engineering notation */
function fmtEng(val) {
  if (val === 0) return '0';
  const abs = Math.abs(val);
  if (abs >= 1e9)  return (val / 1e9).toPrecision(3) + 'G';
  if (abs >= 1e6)  return (val / 1e6).toPrecision(3) + 'M';
  if (abs >= 1e3)  return (val / 1e3).toPrecision(3) + 'k';
  if (abs >= 1)    return val.toPrecision(3);
  if (abs >= 1e-3) return (val * 1e3).toPrecision(3) + 'm';
  if (abs >= 1e-6) return (val * 1e6).toPrecision(3) + '\u00b5';
  if (abs >= 1e-9) return (val * 1e9).toPrecision(3) + 'n';
  if (abs >= 1e-12) return (val * 1e12).toPrecision(3) + 'p';
  return val.toExponential(2);
}

function addWire(wire, color, visRadius) {
  var pts = wire.points;
  var meshes = [];
  if (pts.length < 2) return meshes;

  var mat = new THREE.MeshPhongMaterial({
    color: color, emissive: color, emissiveIntensity: 0.15,
  });

  for (var i = 0; i < pts.length - 1; i++) {
    var p1 = nec(pts[i]);
    var p2 = nec(pts[i + 1]);
    var dir = new THREE.Vector3().subVectors(p2, p1);
    var len = dir.length();
    if (len < 1e-10) continue;

    var cyl = new THREE.Mesh(
      new THREE.CylinderGeometry(visRadius, visRadius, len, 6),
      mat,
    );
    cyl.position.lerpVectors(p1, p2, 0.5);
    cyl.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
    _scene.add(cyl);
    meshes.push(cyl);
  }

  for (var j = 0; j < pts.length; j++) {
    var sph = new THREE.Mesh(new THREE.SphereGeometry(visRadius, 6, 6), mat);
    sph.position.copy(nec(pts[j]));
    _scene.add(sph);
    meshes.push(sph);
  }
  return meshes;
}

function addMarker(position, color, radius, shape) {
  var mat = new THREE.MeshPhongMaterial({
    color: color, emissive: color, emissiveIntensity: 0.4,
    transparent: true, opacity: 0.9,
  });
  var geo;
  if (shape === 'diamond') {
    geo = new THREE.OctahedronGeometry(radius);
  } else if (shape === 'cube') {
    geo = new THREE.BoxGeometry(radius * 1.4, radius * 1.4, radius * 1.4);
  } else if (shape === 'ring') {
    geo = new THREE.TorusGeometry(radius, radius * 0.3, 8, 16);
  } else {
    geo = new THREE.SphereGeometry(radius, 12, 12);
  }
  var mesh = new THREE.Mesh(geo, mat);
  mesh.position.copy(nec(position));
  _scene.add(mesh);
  return mesh;
}

function addAxisArrow(origin, dir, length, color) {
  var arrow = new THREE.ArrowHelper(dir, origin, length, color, length * 0.1, length * 0.05);
  _scene.add(arrow);
}

function makeLabel(text, color, size, bold) {
  var div = document.createElement('div');
  div.textContent = text;
  Object.assign(div.style, {
    color: color || '#e8ecf4',
    fontSize: (size || 11) + 'px',
    fontFamily: 'monospace',
    fontWeight: bold ? '700' : '400',
    background: 'rgba(13,17,23,0.7)',
    padding: '1px 4px',
    borderRadius: '3px',
    whiteSpace: 'nowrap',
    pointerEvents: 'none',
  });
  return new CSS2DObject(div);
}

function addDimensionLine(necPt1, necPt2, label, offsetY) {
  // Project both points down to offsetY for a clean dimension line
  var p1 = new THREE.Vector3(necPt1[0], offsetY, -necPt1[1]);
  var p2 = new THREE.Vector3(necPt2[0], offsetY, -necPt2[1]);
  var wire1 = nec(necPt1);
  var wire2 = nec(necPt2);

  var mat = new THREE.LineDashedMaterial({
    color: COLORS.dimLine, dashSize: 0.15, gapSize: 0.08,
  });

  // Horizontal dimension line
  var lineGeo = new THREE.BufferGeometry().setFromPoints([p1, p2]);
  var line = new THREE.Line(lineGeo, mat);
  line.computeLineDistances();
  _scene.add(line);

  // Vertical leader lines
  var leader = new THREE.LineBasicMaterial({ color: COLORS.dimLine, transparent: true, opacity: 0.3 });
  for (var idx = 0; idx < 2; idx++) {
    var wp = idx === 0 ? wire1 : wire2;
    var dp = idx === 0 ? p1 : p2;
    var lGeo = new THREE.BufferGeometry().setFromPoints([wp, dp]);
    _scene.add(new THREE.Line(lGeo, leader));
  }

  // Invisible hover mesh at midpoint for tooltip
  var dir = new THREE.Vector3().subVectors(p2, p1);
  var len = dir.length();
  if (len < 1e-10) return null;
  var hitR = Math.max(len * 0.03, 0.05);
  var hitMesh = new THREE.Mesh(
    new THREE.CylinderGeometry(hitR, hitR, len, 4),
    new THREE.MeshBasicMaterial({ visible: false }),
  );
  hitMesh.position.lerpVectors(p1, p2, 0.5);
  hitMesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
  _scene.add(hitMesh);
  return { mesh: hitMesh, text: label };
}

function addSpacingLine(mid1, mid2, label) {
  var p1 = nec(mid1);
  var p2 = nec(mid2);

  var mat = new THREE.LineDashedMaterial({
    color: 0x58a6ff, dashSize: 0.2, gapSize: 0.1, transparent: true, opacity: 0.5,
  });
  var lineGeo = new THREE.BufferGeometry().setFromPoints([p1, p2]);
  var line = new THREE.Line(lineGeo, mat);
  line.computeLineDistances();
  _scene.add(line);

  // Invisible hover mesh for tooltip
  var dir = new THREE.Vector3().subVectors(p2, p1);
  var len = dir.length();
  if (len < 1e-10) return null;
  var hitR = Math.max(len * 0.03, 0.05);
  var hitMesh = new THREE.Mesh(
    new THREE.CylinderGeometry(hitR, hitR, len, 4),
    new THREE.MeshBasicMaterial({ visible: false }),
  );
  hitMesh.position.lerpVectors(p1, p2, 0.5);
  hitMesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
  _scene.add(hitMesh);
  return { mesh: hitMesh, text: '\u2194 ' + label };
}


/* ================================================================
   INFO PANEL — collapsible HUD overlay
   ================================================================ */

function buildInfoPanel(geo, wireInfos, spacings) {
  if (!_container) return;
  _infoPanel = document.createElement('div');
  Object.assign(_infoPanel.style, {
    position: 'absolute', top: '8px', left: '8px',
    background: 'rgba(22,26,46,0.92)', color: '#e8ecf4',
    border: '1px solid #2a3050', borderRadius: '8px',
    padding: '0', fontSize: '11px', fontFamily: 'monospace',
    lineHeight: '1.6', zIndex: '50', maxWidth: '260px',
    maxHeight: 'calc(100% - 16px)', overflowY: 'auto',
    overflowX: 'hidden',
  });

  var header = document.createElement('div');
  Object.assign(header.style, {
    padding: '6px 10px', cursor: 'pointer', userSelect: 'none',
    borderBottom: '1px solid #2a3050', display: 'flex',
    alignItems: 'center', justifyContent: 'space-between',
  });
  header.innerHTML = '<span style="font-weight:600;font-size:12px;">Antenna Info</span>' +
                     '<span class="v3d-toggle" style="font-size:10px;opacity:0.6;">\u25bc</span>';

  var body = document.createElement('div');
  body.style.padding = '8px 10px';

  var collapsed = false;
  header.addEventListener('click', function() {
    collapsed = !collapsed;
    body.style.display = collapsed ? 'none' : 'block';
    header.querySelector('.v3d-toggle').textContent = collapsed ? '\u25b6' : '\u25bc';
  });

  var lines = [];

  // Ground
  var gtLabel = geo.ground_type === 'free_space' ? 'Free Space' :
                geo.ground_type === 'perfect' ? 'Perfect Ground' :
                geo.ground_type === 'real' ? 'Real Ground' : geo.ground_type;
  lines.push(sec('Environment'));
  lines.push(kv('Ground', gtLabel));
  if (geo.ground_params && geo.ground_params.epsr) {
    lines.push(kv('\u03b5r', geo.ground_params.epsr));
    lines.push(kv('\u03c3', geo.ground_params.sigma + ' S/m'));
  }
  if (geo.frequency) {
    lines.push(kv('Frequency', geo.frequency + ' MHz'));
    var wavelength = (299.792 / geo.frequency).toFixed(2);
    lines.push(kv('Wavelength', wavelength + ' m'));
  }

  // Wires
  if (wireInfos.length) {
    lines.push(sec('Elements'));
    var wl = geo.frequency ? 299.792 / geo.frequency : null;
    for (var i = 0; i < wireInfos.length; i++) {
      var wi = wireInfos[i];
      var lenStr = wi.length.toFixed(2) + ' m';
      if (wl) lenStr += ' (' + (wi.length / wl).toFixed(3) + '\u03bb)';
      lines.push(kv('Wire ' + wi.tag, lenStr));
    }
  }

  // Spacings
  if (spacings.length) {
    lines.push(sec('Spacing'));
    var wl2 = geo.frequency ? 299.792 / geo.frequency : null;
    for (var i = 0; i < spacings.length; i++) {
      var sp = spacings[i];
      var dStr = sp.distance.toFixed(2) + ' m';
      if (wl2) dStr += ' (' + (sp.distance / wl2).toFixed(3) + '\u03bb)';
      lines.push(kv(sp.wire_a + '\u2194' + sp.wire_b, dStr));
    }
  }

  // Height
  var pMinZ = geo.bounds.min[2], pMaxZ = geo.bounds.max[2];
  if (pMinZ > 0.01 || pMaxZ > 0.01) {
    lines.push(sec('Height'));
    var avgZ = (pMinZ + pMaxZ) / 2;
    var wl3 = geo.frequency ? 299.792 / geo.frequency : null;
    var hStr = avgZ.toFixed(1) + ' m';
    if (wl3) hStr += ' (' + (avgZ / wl3).toFixed(2) + '\u03bb)';
    lines.push(kv('Above z=0', hStr));
  }

  // Excitations
  if (geo.excitations && geo.excitations.length) {
    lines.push(sec('Excitations'));
    for (var i = 0; i < geo.excitations.length; i++) {
      var ex = geo.excitations[i];
      var vMag = Math.sqrt(ex.v_real * ex.v_real + ex.v_imag * ex.v_imag).toFixed(3);
      var vPhase = (Math.atan2(ex.v_imag, ex.v_real) * 180 / Math.PI).toFixed(1);
      lines.push(kv('EX w' + ex.tag + ':s' + ex.segment, vMag + '\u2220' + vPhase + '\u00b0'));
    }
  }

  // Transmission lines
  if (geo.transmission_lines && geo.transmission_lines.length) {
    lines.push(sec('Transmission Lines'));
    for (var i = 0; i < geo.transmission_lines.length; i++) {
      var tl = geo.transmission_lines[i];
      lines.push(kv('w' + tl.tag1 + ':' + tl.seg1 + '\u2192w' + tl.tag2 + ':' + tl.seg2,
                     tl.z0 + '\u03a9'));
    }
  }

  // Loads (non-conductivity only)
  var pointLoads = (geo.loads || []).filter(function(ld) { return ld.ld_type !== 5; });
  if (pointLoads.length) {
    lines.push(sec('Loads'));
    for (var i = 0; i < pointLoads.length; i++) {
      var ld = pointLoads[i];
      var parts = [];
      if (ld.zlr) parts.push('R=' + fmtEng(ld.zlr));
      if (ld.zli) parts.push('L=' + fmtEng(ld.zli));
      if (ld.zlc) parts.push('C=' + fmtEng(ld.zlc));
      lines.push(kv('w' + ld.tag + ' s' + ld.seg_start, ld.type_name +
        (parts.length ? ' ' + parts.join(' ') : '')));
    }
  }

  // Wire conductivity loads
  var condLoads = (geo.loads || []).filter(function(ld) { return ld.ld_type === 5; });
  if (condLoads.length) {
    lines.push(sec('Material'));
    for (var i = 0; i < condLoads.length; i++) {
      var ld = condLoads[i];
      lines.push(kv('Wire ' + ld.tag, fmtEng(ld.zlr) + ' S/m'));
    }
  }

  // Legend
  lines.push(sec('Legend'));
  lines.push(legendItem('#f85149', '\u25c6 Feed point (EX)'));
  lines.push(legendItem('#d29922', '\u25cb Transmission line'));
  lines.push(legendItem('#ff9800', '\u25a0 Load (RLC)'));
  lines.push(legendItem('#58a6ff', '\u2014 Wire'));
  lines.push(legendItem('#d29922', '\u2014 Loaded wire'));

  body.innerHTML = lines.join('');
  _infoPanel.appendChild(header);
  _infoPanel.appendChild(body);
  _container.appendChild(_infoPanel);
}

function sec(title) {
  return '<div style="margin-top:6px;padding-top:4px;border-top:1px solid #2a3050;' +
         'font-weight:600;color:#8b5cf6;font-size:11px;">' + esc(title) + '</div>';
}
function kv(key, val) {
  return '<div style="display:flex;justify-content:space-between;gap:8px;">' +
         '<span style="color:#8b949e;flex-shrink:0;">' + esc(String(key)) + '</span>' +
         '<span style="text-align:right;word-break:break-all;">' + esc(String(val)) + '</span></div>';
}
function legendItem(color, text) {
  return '<div><span style="color:' + color + ';">' + esc(text) + '</span></div>';
}
function esc(s) {
  var d = document.createElement('span');
  d.textContent = s;
  return d.innerHTML;
}

function segMidpoint(points, totalSegments, segIndex) {
  var t = Math.max(0, Math.min(1, (segIndex - 0.5) / Math.max(totalSegments, 1)));

  if (points.length === 2) {
    return [
      points[0][0] + (points[1][0] - points[0][0]) * t,
      points[0][1] + (points[1][1] - points[0][1]) * t,
      points[0][2] + (points[1][2] - points[0][2]) * t,
    ];
  }

  var idx = t * (points.length - 1);
  var lo = Math.floor(idx);
  var hi = Math.min(lo + 1, points.length - 1);
  var frac = idx - lo;
  return [
    points[lo][0] + (points[hi][0] - points[lo][0]) * frac,
    points[lo][1] + (points[hi][1] - points[lo][1]) * frac,
    points[lo][2] + (points[hi][2] - points[lo][2]) * frac,
  ];
}


/* ================================================================
   CURRENT OVERLAY — heat-map coloring by segment magnitude
   ================================================================ */

/**
 * Hot colormap: 0→dark blue, 0.25→cyan, 0.5→green, 0.75→yellow, 1.0→red.
 * Returns a THREE.Color.
 */
function heatColor(t) {
  t = Math.max(0, Math.min(1, t));
  var r, g, b;
  if (t < 0.25) {
    var s = t / 0.25;
    r = 0; g = s; b = 1;
  } else if (t < 0.5) {
    var s = (t - 0.25) / 0.25;
    r = 0; g = 1; b = 1 - s;
  } else if (t < 0.75) {
    var s = (t - 0.5) / 0.25;
    r = s; g = 1; b = 0;
  } else {
    var s = (t - 0.75) / 0.25;
    r = 1; g = 1 - s; b = 0;
  }
  return new THREE.Color(r, g, b);
}

/**
 * Show current magnitude as heat-map overlay on wires.
 * @param {object} data — response from /api/currents endpoint
 *   { ok, by_tag: { "1": { magnitudes: [0..1], phases: [...] }, ... } }
 */
export function showCurrents(data) {
  if (!_scene || !_geometry || !data || !data.by_tag) return;
  clearCurrents();

  // Hide original wire meshes
  for (var entry of _originalWireMeshes) {
    for (var m of entry.meshes) m.visible = false;
  }

  var visRadius = _wireVisRadius * 1.3; // slightly thicker for visibility

  // Build tag→wire lookup from stored geometry
  var wireMap = {};
  for (var wire of (_geometry.wires || [])) {
    wireMap[wire.tag] = wire;
  }

  for (var tagStr in data.by_tag) {
    var wire = wireMap[parseInt(tagStr, 10)];
    if (!wire) continue;
    var mags = data.by_tag[tagStr].magnitudes;
    var nSegs = mags.length;
    if (nSegs === 0 || wire.points.length < 2) continue;

    // Subdivide wire into nSegs equal segments
    // Wire endpoints in NEC coords
    var p0 = wire.points[0];
    var p1 = wire.points[wire.points.length - 1];

    for (var i = 0; i < nSegs; i++) {
      var t0 = i / nSegs;
      var t1 = (i + 1) / nSegs;
      var start = [
        p0[0] + (p1[0] - p0[0]) * t0,
        p0[1] + (p1[1] - p0[1]) * t0,
        p0[2] + (p1[2] - p0[2]) * t0,
      ];
      var end = [
        p0[0] + (p1[0] - p0[0]) * t1,
        p0[1] + (p1[1] - p0[1]) * t1,
        p0[2] + (p1[2] - p0[2]) * t1,
      ];

      var s3 = nec(start);
      var e3 = nec(end);
      var dir = new THREE.Vector3().subVectors(e3, s3);
      var len = dir.length();
      if (len < 1e-10) continue;

      var col = heatColor(mags[i]);
      var mat = new THREE.MeshPhongMaterial({
        color: col, emissive: col, emissiveIntensity: 0.3,
      });
      var cyl = new THREE.Mesh(
        new THREE.CylinderGeometry(visRadius, visRadius, len, 8),
        mat,
      );
      cyl.position.lerpVectors(s3, e3, 0.5);
      cyl.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
      _scene.add(cyl);
      _currentMeshes.push(cyl);
    }
  }
}

/**
 * Remove current overlay and restore original wire colors.
 */
export function clearCurrents() {
  for (var m of _currentMeshes) {
    _scene.remove(m);
    if (m.geometry) m.geometry.dispose();
    if (m.material) m.material.dispose();
  }
  _currentMeshes = [];

  // Restore original wire meshes
  for (var entry of _originalWireMeshes) {
    for (var m of entry.meshes) m.visible = true;
  }
}
