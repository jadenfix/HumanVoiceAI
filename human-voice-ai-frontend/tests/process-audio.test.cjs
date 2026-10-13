const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const test = require('node:test');
const ts = require('typescript');
const filename = path.resolve(__dirname, '../src/app/api/process-audio/route.ts');
const compiled = ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 }
});
const loaded = new Module(filename, module);
loaded.filename = filename;
loaded.paths = Module._nodeModulePaths(path.dirname(filename));
loaded._compile(compiled.outputText, filename);
const { POST } = loaded.exports;

async function send(value) {
  const form = new FormData();
  if (value !== undefined) form.set('audio', value);
  const response = await POST(new Request('http://localhost/api/process-audio', { method: 'POST', body: form }));
  return { response, payload: await response.json() };
}

for (const [name, value] of [['missing', undefined], ['text field', 'not audio'], ['empty file', new Blob([])]]) {
  test(`rejects ${name}`, async () => {
    const { response, payload } = await send(value);
    assert.equal(response.status, 400);
    assert.equal(payload.success, false);
    assert.equal(payload.emotion, undefined);
  });
}

test('rejects malformed multipart input', async () => {
  const response = await POST(new Request('http://localhost/api/process-audio', { method: 'POST', body: 'malformed' }));
  assert.equal(response.status, 400);
  assert.equal((await response.json()).success, false);
});

test('never returns invented emotions, confidence, or processing time for uploaded bytes', async () => {
  const originalFetch = global.fetch;
  let requests = 0;
  global.fetch = async () => { requests++; return new Response('healthy', { status: 200 }); };
  try {
    for (const size of [1, 1000, 4000, 9000]) {
      const { response, payload } = await send(new Blob([new Uint8Array(size)], { type: 'audio/webm' }));
      assert.equal(response.status, 503);
      assert.equal(payload.success, false);
      assert.equal(payload.status, 'inference_unavailable');
      assert.equal(payload.emotion, undefined);
      assert.equal(payload.response, undefined);
      assert.equal(payload.metadata, undefined);
    }
    assert.equal(requests, 0, 'Streamlit health must not be mistaken for inference');
  } finally { global.fetch = originalFetch; }
});
