// Acceptance test for the index.html pitch detector.
// Mirrors the shipped algorithm exactly: McLeod Pitch Method with MPM_K = 0.93,
// first-key-max selection, parabolic interpolation, and the continuous 75 Hz
// high-pass that lives in the audio graph (emulated here with a warmed-up
// biquad so there is no per-buffer startup transient).
//
// Requirement: synthesized sine tones C3..C5 must read within 3 cents.
//
//   node mpm_test.js

var A4 = 440, MIN_FREQ = 70, MAX_FREQ = 1200, RMS_GATE = 0.01;
var CLARITY_GATE = 0.6, MPM_K = 0.93;
var rate = 44100, size = 2048;

// ---- detector (identical logic to index.html detectPitch) ----
function detectPitch(buf, rate) {
  var sizeL = buf.length, i, j;
  var rms = 0;
  for (i = 0; i < sizeL; i++) rms += buf[i] * buf[i];
  rms = Math.sqrt(rms / sizeL);
  if (rms < RMS_GATE) return null;

  var maxLag = Math.ceil(rate / MIN_FREQ) + 2;
  if (maxLag >= sizeL) maxLag = sizeL - 1;
  var nsdf = new Float32Array(maxLag + 1);

  for (var tau = 0; tau <= maxLag; tau++) {
    var acf = 0, div = 0;
    for (i = 0, j = tau; j < sizeL; i++, j++) {
      acf += buf[i] * buf[j];
      div += buf[i] * buf[i] + buf[j] * buf[j];
    }
    nsdf[tau] = div > 0 ? (2 * acf) / div : 0;
  }

  var maxima = [], pos = 1;
  while (pos < maxLag && nsdf[pos] > 0) pos++;
  while (pos < maxLag && nsdf[pos] <= 0) pos++;
  var curMaxLag = -1, curMaxVal = -Infinity;
  while (pos < maxLag) {
    if (nsdf[pos] > nsdf[pos - 1] && nsdf[pos] >= nsdf[pos + 1]) {
      if (nsdf[pos] > curMaxVal) { curMaxVal = nsdf[pos]; curMaxLag = pos; }
    }
    pos++;
    if (pos < maxLag && nsdf[pos] <= 0) {
      if (curMaxLag >= 0) { maxima.push(curMaxLag, curMaxVal); curMaxLag = -1; curMaxVal = -Infinity; }
      while (pos < maxLag && nsdf[pos] <= 0) pos++;
    }
  }
  if (curMaxLag >= 0) maxima.push(curMaxLag, curMaxVal);
  if (!maxima.length) return null;

  var highest = 0;
  for (i = 1; i < maxima.length; i += 2) if (maxima[i] > highest) highest = maxima[i];
  var threshold = MPM_K * highest;

  var chosenLag = -1, chosenVal = 0;
  for (i = 0; i < maxima.length; i += 2) {
    if (maxima[i + 1] >= threshold) { chosenLag = maxima[i]; chosenVal = maxima[i + 1]; break; }
  }
  if (chosenLag < 1 || chosenVal < CLARITY_GATE) return null;

  var refinedLag = chosenLag;
  if (chosenLag > 1 && chosenLag < maxLag) {
    var a = nsdf[chosenLag - 1], b = nsdf[chosenLag], c = nsdf[chosenLag + 1];
    var denom = a - 2 * b + c;
    if (denom !== 0) {
      var shift = 0.5 * (a - c) / denom;
      if (shift > -1 && shift < 1) refinedLag = chosenLag + shift;
    }
  }
  var freq = rate / refinedLag;
  if (freq < MIN_FREQ || freq > MAX_FREQ) return null;
  return freq;
}

// ---- continuous high-pass emulation (matches the graph BiquadFilterNodes) ----
// generate a long signal, filter with persistent state, detect on the tail so
// there is no startup transient - exactly how the live graph filter behaves.
function detectThroughGraph(genLong) {
  var L = 8192;
  var f0 = 75, Q = 0.707;
  var w0 = 2 * Math.PI * f0 / rate, cw = Math.cos(w0), sw = Math.sin(w0), alpha = sw / (2 * Q);
  var b0 = (1 + cw) / 2, b1 = -(1 + cw), b2 = (1 + cw) / 2, a0 = 1 + alpha, a1 = -2 * cw, a2 = 1 - alpha;
  b0 /= a0; b1 /= a0; b2 /= a0; a1 /= a0; a2 /= a0;
  var cur = genLong(L);
  for (var s = 0; s < 2; s++) {                 // two cascaded stages
    var out = new Float32Array(L), x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    for (var i = 0; i < L; i++) {
      var x = cur[i];
      var y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
      out[i] = y; x2 = x1; x1 = x; y2 = y1; y1 = y;
    }
    cur = out;
  }
  return detectPitch(cur.subarray(L - size), rate);
}

// ---- helpers ----
function midiToFreq(m) { return A4 * Math.pow(2, (m - 69) / 12); }
function cents(f, ref) { return 1200 * Math.log2(f / ref); }

var failures = 0;

// ===== required acceptance test: pure sines C3..C5 within 3 cents =====
console.log('=== pure sine tones, C3..C5 (must be within 3 cents) ===');
for (var m = 48; m <= 72; m++) {
  var f = midiToFreq(m);
  var buf = new Float32Array(size);
  for (var n = 0; n < size; n++) buf[n] = Math.sin(2 * Math.PI * f * n / rate);
  var det = detectPitch(buf, rate);
  var err = det ? cents(det, f) : NaN;
  var ok = det && Math.abs(err) <= 3;
  if (!ok) failures++;
  console.log(
    (ok ? '  ok  ' : ' FAIL ') + 'midi ' + m + '  ' + f.toFixed(2) + ' Hz -> ' +
    (det ? det.toFixed(2) + ' Hz  ' + (err >= 0 ? '+' : '') + err.toFixed(2) + ' c' : 'null')
  );
}

// ===== robustness: full graph (high-pass + detector) under DC + rumble =====
// real-mic conditions that previously caused the consistent sharp bias.
console.log('\n=== through audio graph: DC offset + sub-bass rumble (within 3 cents) ===');
var SCENARIOS = [
  ['DC 0.25 + 45Hz x0.8', { dc: 0.25, r: [[0.8, 45]] }],
  ['30Hz x0.5',           { dc: 0,    r: [[0.5, 30]] }],
  ['20Hz x0.8 + DC 0.2',  { dc: 0.2,  r: [[0.8, 20]] }]
];
SCENARIOS.forEach(function (sc) {
  var worst = 0, worstM = 0;
  for (var mm = 48; mm <= 72; mm++) {
    var ff = midiToFreq(mm);
    var det2 = detectThroughGraph((function (fund, opt) {
      return function (L) {
        var b = new Float32Array(L);
        for (var k = 0; k < L; k++) {
          var t = k / rate;
          var v = opt.dc + Math.sin(2 * Math.PI * fund * t) +
                  0.6 * Math.sin(2 * Math.PI * 2 * fund * t) +
                  0.4 * Math.sin(2 * Math.PI * 3 * fund * t);
          opt.r.forEach(function (p) { v += p[0] * Math.sin(2 * Math.PI * p[1] * t); });
          b[k] = v;
        }
        return b;
      };
    })(ff, sc[1]));
    var e = det2 ? cents(det2, ff) : NaN;
    if (!det2) { worst = NaN; break; }
    if (Math.abs(e) > Math.abs(worst)) { worst = e; worstM = mm; }
  }
  var ok = isFinite(worst) && Math.abs(worst) <= 3;
  if (!ok) failures++;
  console.log((ok ? '  ok  ' : ' FAIL ') + sc[0] + ': worst ' +
    (isFinite(worst) ? (worst >= 0 ? '+' : '') + worst.toFixed(2) + ' c (midi ' + worstM + ')' : 'null'));
});

// ===== octave-error guard: weak fundamental + strong harmonics must NOT jump =====
console.log('\n=== weak fundamental + strong harmonics (no octave jump) ===');
var octFail = 0;
for (var mo = 48; mo <= 72; mo++) {
  var fo = midiToFreq(mo);
  var bo = new Float32Array(size);
  for (var no = 0; no < size; no++) {
    var to = no / rate;
    bo[no] = 0.12 * Math.sin(2 * Math.PI * fo * to)        // very weak fundamental
           + 1.00 * Math.sin(2 * Math.PI * 2 * fo * to)     // strong octave harmonic
           + 0.85 * Math.sin(2 * Math.PI * 3 * fo * to)
           + 0.45 * Math.sin(2 * Math.PI * 4 * fo * to);
  }
  var dor = detectPitch(bo, rate);
  var eo = dor ? cents(dor, fo) : NaN;
  if (!dor || Math.abs(eo) > 50) { octFail++; failures++;
    console.log(' FAIL midi ' + mo + ' -> ' + (dor ? dor.toFixed(2) + ' Hz (' + eo.toFixed(0) + ' c)' : 'null')); }
}
if (!octFail) console.log('  ok  no octave jumps across C3..C5');

console.log('\n' + (failures === 0 ? 'ALL TESTS PASSED' : failures + ' TEST(S) FAILED'));
process.exit(failures === 0 ? 0 : 1);
