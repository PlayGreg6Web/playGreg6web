// Sanitized crash reporter and runtime diagnostic bundle exporter for playgta6web.
// Zero external dependencies; clean ES module compatible with <script type="module">
// and Node.js ESM imports. Redacts local host paths, file:// URLs, query strings,
// and auth tokens so exported diagnostic bundles never leak host usernames,
// local filesystem paths, secrets, or game data.

export const SCHEMA_VERSION = 1;
export const DEFAULT_MAX_LOGS = 200;
export const DEFAULT_MAX_ERRORS = 32;
export const DEFAULT_APP_VERSION = '0.1.0';
export const MAX_GPU_ADAPTER_LENGTH = 96;

export const ERROR_CATEGORIES = Object.freeze([
  'wasm_trap',
  'webgpu_validation',
  'webgpu_device_lost',
  'worker_error',
  'io_error',
  'unhandled_rejection',
  'runtime',
]);

export const BOOT_ERROR_CODES = Object.freeze({
  WASM_MODULE_TOO_LARGE: 'WASM_MODULE_TOO_LARGE',
  WASM64_UNSUPPORTED: 'WASM64_UNSUPPORTED',
  COI_SAB_MISSING: 'COI_SAB_MISSING',
  WORKER_SCRIPT_COLLISION: 'WORKER_SCRIPT_COLLISION',
  HTTPFS_PGIM_ERROR: 'HTTPFS_PGIM_ERROR',
  UNKNOWN_BOOT_ERROR: 'UNKNOWN_BOOT_ERROR',
});

export const LOG_LEVELS = Object.freeze(['debug', 'info', 'ok', 'warn', 'error']);

const ERROR_CATEGORY_SET = new Set(ERROR_CATEGORIES);
const LOG_LEVEL_SET = new Set(LOG_LEVELS);
const MAX_SANITIZE_DEPTH = 16;
const MAX_BOOT_MILESTONES = 32;
const MAX_BOOT_TRAPS = 32;
const MAX_BOOT_UNRESOLVED_IMPORTS = 64;
const MAX_GPU_MISSING_SHADERS = 64;
const MAX_ERROR_MESSAGE_LENGTH = 4096;
const MAX_ERROR_STACK_LENGTH = 16384;
const MAX_LOG_MESSAGE_LENGTH = 4096;

const WASM_MODULE_TOO_LARGE_RE =
  /\b(?:exceeds\s+maximum\s+size\s+of\s+1073741824|buffer\s+source\s+exceeds\s+maximum\s+size|(?:wasm\s+)?module\s+(?:size\s+)?exceeds\s+(?:the\s+)?(?:1\s*gib|1073741824|maximum\s+size))\b/i;

const WASM64_UNSUPPORTED_RE =
  /\b(?:wasm64|memory64)\b[^.\n]{0,80}\b(?:unsupported|not\s+supported|not\s+enabled|unavailable|disabled|missing|required|failed)\b|\b(?:unsupported|not\s+supported|does\s+not\s+support|unavailable|disabled|missing|no|lacks?)\b[^.\n]{0,80}\b(?:wasm64|memory64)\b|\b(?:invalid|unsupported|unknown)\s+(?:memory\s+)?(?:address|index)(?:\s+type)?\s*[:=]?\s*['"]?i64['"]?|\bi64\s+memory\s+(?:address|index)\b[^.\n]{0,40}\b(?:unsupported|not\s+supported|disabled|invalid)\b/i;

const COI_SAB_MISSING_RE =
  /\bsharedarraybuffer\b[^.\n]{0,80}\b(?:not\s+defined|undefined|missing|unavailable|disabled|unsupported|not\s+supported|requires?\s+cross[\s-]*origin|cannot\s+be\s+(?:created|used|transferred|cloned))\b|\b(?:missing|no|lacks?|requires?|without|not)\b[^.\n]{0,80}\b(?:sharedarraybuffer|cross[\s-]*origin[\s-]*isolat(?:ed|ion)|coop\s*\/\s*coep)\b|\bcross[\s-]*origin[\s-]*isolat(?:ed|ion)\b[^.\n]{0,80}\b(?:false|missing|required|disabled|unavailable|not\s+enabled|not\s+active)\b|\b(?:coop\s*\/\s*coep|cross-origin-(?:opener|embedder)-policy)\b/i;

const WORKER_SCRIPT_COLLISION_RE =
  /\bhas\s+already\s+been\s+declared\b|\bredeclaration\s+of\s+(?:const|let|var|class|function)\b/i;

const HTTPFS_PGIM_RE = /\b(?:httpfs|pgim)\b|\.pgim\b/i;

const QUOTED_FILE_URL_RE = /(["'`])(file:\/\/[^"'`\r\n]+?)(:\d+:\d+)?\1/gi;

const FILE_URL_RE =
  /\bfile:\/\/[^\s"'`<>)\]}]+?(?=:\d+:\d+(?:[)\s"'`<>\]};,]|$)|[\s"'`<>)\]}]|$)/gi;

const URL_USERINFO_RE = /\b((?:https?|wss?|ftps?|sftp|smb|ssh):\/\/)([^\s/@"'`<>]+)@/gi;

const URL_QUERY_RE =
  /((?:https?|wss?|ftp):\/\/[^\s"'`<>?#]+)\?(?!\[REDACTED\])[^\s"'`<>#)\]}]+?(?=:\d+:\d+(?:[)\s"'`<>\]}]|$)|[\s"'`<>#)\]}]|$)/gi;

const STANDALONE_QUERY_RE =
  /\?[A-Za-z0-9_.~%-]+=[^\s"'`<>#)\]}]+?(?=:\d+:\d+(?:[)\s"'`<>\]}]|$)|[\s"'`<>#)\]}]|$)/g;

const AUTH_SCHEME_RE = /\b(Bearer|Basic)\s+(?!\[REDACTED\])[A-Za-z0-9\-._~+/=:]+/gi;

const PEM_PRIVATE_KEY_RE =
  /-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----/g;

const SECRET_KV_RE =
  /\b(authorization|api[_-]?key|apikey|api[_-]?secret|access[_-]?key|access[_-]?token|refresh[_-]?token|auth[_-]?token|id[_-]?token|oauth[_-]?token|bearer[_-]?token|github[_-]?token|npm[_-]?token|client[_-]?secret|private[_-]?key|secret[_-]?key|signing[_-]?key|secret[_-]?token|session[_-]?token|session[_-]?id|csrf[_-]?token|x-csrf-token|sas[_-]?token|aws[_-]?secret[_-]?access[_-]?key|aws[_-]?access[_-]?key[_-]?id|aws[_-]?session[_-]?token|x-api-key|x-auth-token|x-access-token|x-session-token|x-amz-signature|x-amz-security-token|x-amz-credential|cookie|set-cookie|password|passwd|secret|token|jwt|username|user[_-]?name|os[_-]?user|host[_-]?user|user[_-]?profile|home[_-]?dir|homedir)(["']?\s*[:=]\s*)(?!(?:(?:Bearer|Basic)\s+)?\[REDACTED\])(?:"[^"]*"|'[^']*'|[^\s,;}\])&]+)/gi;

const BARE_TOKEN_RE =
  /\b(?:eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}|(?:gh[pousr]|github_pat)_[A-Za-z0-9_]{16,}|sk_(?:live|test)_[A-Za-z0-9]{12,}|sk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}|(?:AKIA|ASIA)[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z_-]{35,})\b/g;

const DATA_URI_BASE64_RE =
  /\bdata:[a-z0-9.+-]+\/[a-z0-9.+-]+(?:;[^,\s"'`<>]*)?;base64,[A-Za-z0-9+/=\r\n_-]{16,}/gi;

const HEX_DUMP_BYTES_RE =
  /(?<![0-9a-fA-F])(?:[0-9a-fA-F]{2}[ \t]+){16,}[0-9a-fA-F]{2}(?![0-9a-fA-F])/g;

const LONG_HEX_PAYLOAD_RE = /(?<![0-9a-fA-F])(?:0x)?[0-9a-fA-F]{65,}(?![0-9a-fA-F])/g;

const BINARY_CONTROL_CHAR_RE = /[\x00-\x08\x0b\x0c\x0e-\x1f]/;
const BINARY_CONTROL_RUN_RE = /[^\r\n\t\x20-\x7e]*[\x00-\x08\x0b\x0c\x0e-\x1f][^\r\n\t\x20-\x7e]*/g;

const QUOTED_WIN_PATH_RE =
  /(["'`])((?:\\{2,}[?.]\\{1,2})?[A-Za-z]:(?:\\{1,2}|\/)[^"'`\r\n]+?)(:\d+:\d+)?\1/g;

const QUOTED_UNC_PATH_RE =
  /(["'`])(\\{2,}(?:[?.]\\{1,2})?[^\s\\"'`\r\n][^"'`\r\n]*?\\[^"'`\r\n]+?)(:\d+:\d+)?\1/g;

const QUOTED_FWD_UNC_PATH_RE =
  /(["'`])(\/\/[A-Za-z0-9._-]+\/[^"'`\r\n]+?)(:\d+:\d+)?\1/g;

const WIN_DEVICE_PATH_RE =
  /(^|[\s"'`(\[={<,:;|])(\\{2,}[?.]\\+[^\s"'`<>|)\]},;]+?)(?=:\d+:\d+(?:[)\s"'`<>\]};,]|$)|[\s"'`<>|)\]},;]|$)/g;

const WIN_DRIVE_PATH_RE =
  /(?<![A-Za-z0-9_])([A-Za-z]:(?:\\|\/(?!\/))[^\s"'`<>|*?)\]},;]+?)(?=:\d+:\d+(?:[)\s"'`<>\]};,]|$)|[\s"'`<>|*?)\]},;]|$)/g;

const WIN_UNC_PATH_RE =
  /(^|[\s"'`(\[={<,:;|])(\\{2,}[^\s\\"'`<>|*?)\]},;]+\\+[^\s"'`<>|*?)\]},;]+?)(?=:\d+:\d+(?:[)\s"'`<>\]};,]|$)|[\s"'`<>|*?)\]},;]|$)/g;

const FWD_UNC_PATH_RE =
  /(^|[\s"'`(\[={<,;|])(?<![A-Za-z0-9_:/])(\/\/[A-Za-z0-9._-]+\/[^\s/"'`<>)\]},;]+(?:\/[^\s"'`<>)\]},;]+)*)(?=:\d+:\d+(?:[)\s"'`<>\]};,]|$)|[\s"'`<>)\]},;]|$)/g;

const POSIX_ROOTS =
  'home|Home|Users|users|root|tmp|var|etc|opt|usr|mnt|media|Volumes|volumes|private|srv|proc|sys|dev|run|Library|library|Applications|applications|cygdrive|wsl\\.localhost|wsl\\$';

const QUOTED_POSIX_PATH_RE = new RegExp(
  `(["'\`])(\\/(?:${POSIX_ROOTS})(?:\\/[^"'\`\\r\\n]*?)?)(:\\d+:\\d+)?\\1`,
  'g'
);

const POSIX_ABS_PATH_RE = new RegExp(
  `(^|[\\s"'\`(\\[={<,:;|])(\\/(?:${POSIX_ROOTS})(?:\\/[^\\s"'\`<>\\)\\]},;]+?)?)(?=:\\d+:\\d+(?:[)\\s"'\`<>\\]};,]|$)|[\\s"'\`<>\\)\\]},;]|$)`,
  'g'
);

const MSYS_DRIVE_USER_PATH_RE =
  /(^|[\s"'`(\[={<,:;|])(\/[a-zA-Z]\/(?:Users|Windows|Program(?:\s+Files)?|Documents\s+and\s+Settings)(?:\/[^\s"'`<>)\]},;]+?)?)(?=:\d+:\d+(?:[)\s"'`<>\]};,]|$)|[\s"'`<>)\]},;]|$)/gi;

const QUOTED_TILDE_HOME_PATH_RE = /(["'`])(~[A-Za-z0-9._-]*\/[^"'`\r\n]+?)(:\d+:\d+)?\1/g;

const TILDE_HOME_PATH_RE =
  /(^|[\s"'`(\[={<,:;|])(~[A-Za-z0-9._-]*\/[^\s"'`<>)\]},;]+?)(?=:\d+:\d+(?:[)\s"'`<>\]};,]|$)|[\s"'`<>)\]},;]|$)/g;

const SENSITIVE_KEY_RE =
  /^(?:authorization|cookie|set[_-]?cookie|password|passwd|secret|client[_-]?secret|api[_-]?key|apikey|api[_-]?secret|access[_-]?key|access[_-]?token|refresh[_-]?token|auth[_-]?token|id[_-]?token|oauth[_-]?token|bearer[_-]?token|github[_-]?token|npm[_-]?token|session[_-]?token|session[_-]?id|csrf[_-]?token|x[_-]?csrf[_-]?token|sas[_-]?token|private[_-]?key|secret[_-]?key|signing[_-]?key|secret[_-]?token|aws[_-]?(?:secret[_-]?access[_-]?key|access[_-]?key[_-]?id|session[_-]?token)|x[_-]?api[_-]?key|x[_-]?auth[_-]?token|x[_-]?access[_-]?token|x[_-]?session[_-]?token|x[_-]?amz[_-]?(?:signature|security[_-]?token|credential)|jwt|token|username|user[_-]?name|os[_-]?user|host[_-]?user|user[_-]?profile|home[_-]?dir|homedir)$/i;

const MAX_SANITIZE_CLEAN_CACHE = 512;
const MAX_SANITIZE_CACHE_STRING_LENGTH = 256;
const _cleanSanitizeCache = new Set();

/**
 * Strips Windows paths (`C:\Users\...`), UNC paths (`\\server\share\...`, `//server/share/...`),
 * POSIX host paths (`/home/...`, `/Users/...`), `file://` URLs, query strings (`?token=...`),
 * URL credentials, bearer/auth tokens, usernames, and raw binary payloads (data URIs, multi-byte
 * hex dumps, binary control bytes) while preserving sanitized hex VAs (`0x...`), instruction
 * offsets, and shader hashes.
 *
 * @param {unknown} text
 * @returns {string}
 */
export function sanitizeString(text) {
  if (text === null || text === undefined) {
    return '';
  }

  let out = typeof text === 'string' ? text : String(text);
  const origLen = out.length;
  if (origLen === 0) {
    return out;
  }

  // Fast O(1) cache hit for repeated clean strings (never stores sensitive/redacted inputs).
  if (origLen <= MAX_SANITIZE_CACHE_STRING_LENGTH && _cleanSanitizeCache.has(out)) {
    return out;
  }

  // Single-pass character class scan so clean telemetry/UI strings skip all irrelevant regexes
  // while preserving 100% identical redaction behavior on every path, URL, secret, and buffer.
  let hasControl = false;
  let hasColon = false;
  let hasEquals = false;
  let hasSlash = false;
  let hasBackslash = false;
  let hasQuestion = false;
  let hasTilde = false;
  let hasAt = false;
  let hasDash = false;
  let hasUnderscore = false;
  let hasDot = false;
  let hasWhitespace = false;
  let hasBChar = false;
  let hasUpperA = false;

  for (let i = 0; i < origLen; i += 1) {
    const c = out.charCodeAt(i);
    if (c < 0x20) {
      if (c === 0x09 || c === 0x0a || c === 0x0d) {
        hasWhitespace = true;
      } else {
        hasControl = true;
      }
    } else if (c === 0x20 || c >= 0xa0) {
      hasWhitespace = true;
    } else if (c === 0x3a) {
      hasColon = true;
    } else if (c === 0x3d) {
      hasEquals = true;
    } else if (c === 0x2f) {
      hasSlash = true;
    } else if (c === 0x5c) {
      hasBackslash = true;
    } else if (c === 0x3f) {
      hasQuestion = true;
    } else if (c === 0x7e) {
      hasTilde = true;
    } else if (c === 0x40) {
      hasAt = true;
    } else if (c === 0x2d) {
      hasDash = true;
    } else if (c === 0x5f) {
      hasUnderscore = true;
    } else if (c === 0x2e) {
      hasDot = true;
    } else if (c === 0x42 || c === 0x62) {
      hasBChar = true;
    } else if (c === 0x41) {
      hasUpperA = true;
    }
  }

  const original = out;

  // 0. Redact raw binary payloads (NUL/control-byte blobs, base64 data URIs, hex dumps).
  if (hasControl) {
    let nulCount = 0;
    let ctrlCount = 0;
    for (let i = 0; i < out.length; i += 1) {
      const code = out.charCodeAt(i);
      if (code === 0) {
        nulCount += 1;
        ctrlCount += 1;
      } else if ((code < 0x20 && code !== 0x09 && code !== 0x0a && code !== 0x0d) || code === 0x7f) {
        ctrlCount += 1;
      }
    }
    if (out.startsWith('\x00asm') || (nulCount >= 2 && ctrlCount / out.length >= 0.1) || ctrlCount >= 8) {
      return '[REDACTED_BUFFER]';
    }
    out = out.replace(BINARY_CONTROL_RUN_RE, '[REDACTED_BUFFER]');
  }
  if (hasColon && hasSlash && origLen >= 30) {
    out = out.replace(DATA_URI_BASE64_RE, '[REDACTED_BUFFER]');
  }
  if (origLen >= 50 && hasWhitespace) {
    out = out.replace(HEX_DUMP_BYTES_RE, '[REDACTED_BUFFER]');
  }
  if (origLen >= 65) {
    out = out.replace(LONG_HEX_PAYLOAD_RE, '[REDACTED_BUFFER]');
  }

  // 1. Redact file:// URLs completely (quoted first to catch spaces, preserving optional :line:col).
  if (hasColon && hasSlash) {
    out = out.replace(QUOTED_FILE_URL_RE, (_, quote, _url, lineCol) => {
      return `${quote}[REDACTED_FILE_URL]${lineCol || ''}${quote}`;
    });
    out = out.replace(FILE_URL_RE, '[REDACTED_FILE_URL]');
  }

  // 2. Redact embedded URL credentials (scheme://user:pass@host and scheme://user@host).
  if (hasAt && hasColon && hasSlash) {
    out = out.replace(URL_USERINFO_RE, '$1[REDACTED]@');
  }

  // 3. Redact query strings on HTTP/WS/FTP URLs and standalone ?key=value tokens.
  if (hasQuestion) {
    if (hasColon && hasSlash) {
      out = out.replace(URL_QUERY_RE, '$1?[REDACTED]');
    }
    if (hasEquals) {
      out = out.replace(STANDALONE_QUERY_RE, '?[REDACTED]');
    }
  }

  // 4. Redact PEM private keys, Bearer/Basic tokens, inline credential/username pairs, and bare JWTs/PATs.
  if (hasDash && origLen >= 46 && out.includes('-----BEGIN ')) {
    out = out.replace(PEM_PRIVATE_KEY_RE, '[REDACTED_TOKEN]');
  }
  if (hasBChar && hasWhitespace && origLen >= 7) {
    out = out.replace(AUTH_SCHEME_RE, '$1 [REDACTED]');
  }
  if (hasColon || hasEquals) {
    out = out.replace(SECRET_KV_RE, '$1$2[REDACTED]');
  }
  if (origLen >= 15 && (hasDot || hasUnderscore || hasDash || hasUpperA)) {
    out = out.replace(BARE_TOKEN_RE, '[REDACTED_TOKEN]');
  }

  // 5. Redact Windows absolute drive, extended-length/device, and UNC paths (quoted first to catch spaces).
  if (hasColon && (hasBackslash || hasSlash)) {
    out = out.replace(QUOTED_WIN_PATH_RE, (_, quote, _path, lineCol) => {
      return `${quote}[REDACTED_PATH]${lineCol || ''}${quote}`;
    });
  }
  if (hasBackslash) {
    out = out.replace(QUOTED_UNC_PATH_RE, (_, quote, _path, lineCol) => {
      return `${quote}[REDACTED_PATH]${lineCol || ''}${quote}`;
    });
  }
  if (hasSlash && out.includes('//')) {
    out = out.replace(QUOTED_FWD_UNC_PATH_RE, (_, quote, _path, lineCol) => {
      return `${quote}[REDACTED_PATH]${lineCol || ''}${quote}`;
    });
  }
  if (hasBackslash) {
    out = out.replace(WIN_DEVICE_PATH_RE, '$1[REDACTED_PATH]');
  }
  if (hasColon && (hasBackslash || hasSlash)) {
    out = out.replace(WIN_DRIVE_PATH_RE, '[REDACTED_PATH]');
  }
  if (hasBackslash) {
    out = out.replace(WIN_UNC_PATH_RE, '$1[REDACTED_PATH]');
  }
  if (hasSlash && out.includes('//')) {
    out = out.replace(FWD_UNC_PATH_RE, '$1[REDACTED_PATH]');
  }

  // 6. Redact POSIX host absolute paths, MSYS drive user paths, and ~ home paths (preserving app routes like /api/status).
  if (hasSlash) {
    out = out.replace(QUOTED_POSIX_PATH_RE, (_, quote, _path, lineCol) => {
      return `${quote}[REDACTED_PATH]${lineCol || ''}${quote}`;
    });
    out = out.replace(POSIX_ABS_PATH_RE, '$1[REDACTED_PATH]');
    out = out.replace(MSYS_DRIVE_USER_PATH_RE, '$1[REDACTED_PATH]');
    if (hasTilde) {
      out = out.replace(QUOTED_TILDE_HOME_PATH_RE, (_, quote, _path, lineCol) => {
        return `${quote}[REDACTED_PATH]${lineCol || ''}${quote}`;
      });
      out = out.replace(TILDE_HOME_PATH_RE, '$1[REDACTED_PATH]');
    }
  }

  if (out === original && origLen <= MAX_SANITIZE_CACHE_STRING_LENGTH) {
    if (_cleanSanitizeCache.size >= MAX_SANITIZE_CLEAN_CACHE) {
      _cleanSanitizeCache.clear();
    }
    _cleanSanitizeCache.add(out);
  }

  return out;
}

function isBinaryBuffer(value) {
  if (!value || typeof value !== 'object') {
    return false;
  }
  if (ArrayBuffer.isView(value) || value instanceof ArrayBuffer) {
    return true;
  }
  if (typeof SharedArrayBuffer === 'function' && value instanceof SharedArrayBuffer) {
    return true;
  }
  return false;
}

function isSerializedNodeBuffer(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return false;
  }
  if (value.type !== 'Buffer' || !Array.isArray(value.data)) {
    return false;
  }
  const keys = Object.keys(value);
  return keys.length === 2 && (value.data.length === 0 || typeof value.data[0] === 'number');
}

function isBlobLike(value) {
  return Boolean(value && typeof value === 'object' && typeof Blob === 'function' && value instanceof Blob);
}

function isErrorLike(value) {
  if (!value || typeof value !== 'object') {
    return false;
  }
  if (value instanceof Error) {
    return true;
  }
  const tag = Object.prototype.toString.call(value);
  return tag === '[object Error]' || tag === '[object DOMException]';
}

/**
 * Recursively sanitizes any value for safe JSON diagnostic serialization.
 * Strips paths/secrets from all strings (and object keys), summarizes binary buffers
 * without dumping raw bytes, handles BigInt/Error/Date/Map/Set, and guards against cycles.
 *
 * @param {unknown} value
 * @param {WeakSet<object>} [seen]
 * @param {number} [depth]
 * @returns {any}
 */
export function sanitizeValue(value, seen = new WeakSet(), depth = 0) {
  if (value === null || value === undefined) {
    return null;
  }

  const kind = typeof value;
  if (kind === 'string') {
    return sanitizeString(value);
  }
  if (kind === 'number') {
    return Number.isFinite(value) ? value : null;
  }
  if (kind === 'boolean') {
    return value;
  }
  if (kind === 'bigint') {
    return value.toString();
  }
  if (kind === 'symbol' || kind === 'function') {
    return undefined;
  }

  if (depth >= MAX_SANITIZE_DEPTH) {
    return '[MaxDepth]';
  }

  if (value instanceof Date) {
    return Number.isNaN(value.getTime()) ? null : value.toISOString();
  }

  // Never dump raw binary buffers (e.g. game assets or Wasm memory slices) into crash reports.
  if (isBinaryBuffer(value)) {
    return {
      type: value.constructor ? sanitizeString(value.constructor.name) : 'BinaryBuffer',
      byteLength: typeof value.byteLength === 'number' ? value.byteLength : 0,
    };
  }
  if (isSerializedNodeBuffer(value)) {
    return {
      type: 'Buffer',
      byteLength: value.data.length,
    };
  }
  if (isBlobLike(value)) {
    return {
      type: 'Blob',
      byteLength: typeof value.size === 'number' && Number.isFinite(value.size) ? value.size : 0,
    };
  }

  if (seen.has(value)) {
    return '[Circular]';
  }
  seen.add(value);

  try {
    if (Array.isArray(value)) {
      const out = [];
      for (let i = 0; i < value.length; i += 1) {
        const sanitized = sanitizeValue(value[i], seen, depth + 1);
        out.push(sanitized === undefined ? null : sanitized);
      }
      return out;
    }

    if (value instanceof Set) {
      const out = [];
      for (const item of value) {
        const sanitized = sanitizeValue(item, seen, depth + 1);
        out.push(sanitized === undefined ? null : sanitized);
      }
      return out;
    }

    if (value instanceof Map) {
      const out = {};
      for (const [k, v] of value.entries()) {
        const rawKey = typeof k === 'string' ? k : String(k);
        const safeKey = sanitizeString(rawKey);
        if (SENSITIVE_KEY_RE.test(rawKey) && typeof v === 'string' && v.length > 0) {
          out[safeKey] = '[REDACTED]';
        } else {
          const sanitized = sanitizeValue(v, seen, depth + 1);
          if (sanitized !== undefined) {
            out[safeKey] = sanitized;
          }
        }
      }
      return out;
    }

    if (isErrorLike(value)) {
      const errOut = {
        name: sanitizeString(value.name || 'Error'),
        message: sanitizeString(value.message || ''),
        stack: typeof value.stack === 'string' ? sanitizeString(value.stack) : null,
      };
      if (value.cause !== undefined) {
        errOut.cause = sanitizeValue(value.cause, seen, depth + 1);
      }
      if (typeof value.code === 'string' || typeof value.code === 'number') {
        errOut.code = sanitizeValue(value.code, seen, depth + 1);
      }
      if (typeof value.reason === 'string') {
        errOut.reason = sanitizeString(value.reason);
      }
      return errOut;
    }

    const out = {};
    for (const key of Object.keys(value)) {
      const raw = value[key];
      if (raw === undefined || typeof raw === 'function' || typeof raw === 'symbol') {
        continue;
      }
      const safeKey = sanitizeString(key);
      if (SENSITIVE_KEY_RE.test(key) && typeof raw === 'string' && raw.length > 0) {
        out[safeKey] = '[REDACTED]';
        continue;
      }
      const sanitized = sanitizeValue(raw, seen, depth + 1);
      if (sanitized !== undefined) {
        out[safeKey] = sanitized;
      }
    }
    return out;
  } finally {
    seen.delete(value);
  }
}

function normalizeLogLevel(level) {
  if (typeof level !== 'string') {
    return 'info';
  }
  const lower = level.trim().toLowerCase();
  if (LOG_LEVEL_SET.has(lower)) {
    return lower;
  }
  if (lower === 'warning') {
    return 'warn';
  }
  if (lower === 'err' || lower === 'fatal') {
    return 'error';
  }
  return 'info';
}

function formatIsoTimestamp(ms) {
  if (typeof ms === 'number' && Number.isFinite(ms)) {
    const date = new Date(ms);
    if (!Number.isNaN(date.getTime())) {
      return date.toISOString();
    }
  }
  return new Date(0).toISOString();
}

/**
 * Classifies an error into one of the canonical diagnostic categories.
 * If `preferredCategory` is already a canonical category, it is returned directly.
 *
 * @param {unknown} errorOrMessage
 * @param {string|null} [preferredCategory]
 * @returns {string}
 */
export function classifyError(errorOrMessage, preferredCategory = null) {
  if (typeof preferredCategory === 'string') {
    const normalized = preferredCategory.trim().toLowerCase();
    if (ERROR_CATEGORY_SET.has(normalized)) {
      return normalized;
    }
    if (normalized === 'unhandledrejection') {
      return 'unhandled_rejection';
    }
  }

  if (
    typeof WebAssembly !== 'undefined' &&
    errorOrMessage &&
    typeof errorOrMessage === 'object' &&
    ((typeof WebAssembly.RuntimeError === 'function' && errorOrMessage instanceof WebAssembly.RuntimeError) ||
      (typeof WebAssembly.CompileError === 'function' && errorOrMessage instanceof WebAssembly.CompileError) ||
      (typeof WebAssembly.LinkError === 'function' && errorOrMessage instanceof WebAssembly.LinkError))
  ) {
    return 'wasm_trap';
  }

  const name =
    errorOrMessage && typeof errorOrMessage === 'object' && typeof errorOrMessage.name === 'string'
      ? errorOrMessage.name
      : '';
  const message =
    typeof errorOrMessage === 'string'
      ? errorOrMessage
      : errorOrMessage && typeof errorOrMessage === 'object' && typeof errorOrMessage.message === 'string'
        ? errorOrMessage.message
        : '';
  const stack =
    errorOrMessage && typeof errorOrMessage === 'object' && typeof errorOrMessage.stack === 'string'
      ? errorOrMessage.stack
      : '';
  const combined = `${name} ${message} ${stack}`;

  if (/\b(?:device[\s_-]*lost|gpudevicelostinfo|webgpu.*lost)\b/i.test(combined)) {
    return 'webgpu_device_lost';
  }
  if (
    /\b(?:gpuvalidationerror|gpuoutofmemoryerror|gpuinternalerror|webgpu|wgpu|wgsl|validation\s+error)\b/i.test(
      combined
    )
  ) {
    return 'webgpu_validation';
  }
  if (
    /\b(?:wasm|webassembly|unreachable|memory\s+access\s+out\s+of\s+bounds|out\s+of\s+bounds\s+memory|indirect\s+call|integer\s+divide\s+by\s+zero|wasm_trap)\b/i.test(
      combined
    )
  ) {
    return 'wasm_trap';
  }
  if (/\b(?:worker|importscripts|postmessage|offscreencanvas)\b/i.test(combined)) {
    return 'worker_error';
  }
  if (/\b(?:opfs|httpfs|range\s+request|storage|quota|fetch|network|manifest|bootset|io_error)\b/i.test(combined)) {
    return 'io_error';
  }
  if (/\bunhandled\s*rejection\b/i.test(combined)) {
    return 'unhandled_rejection';
  }

  return 'runtime';
}

function extractRawBootErrorText(errorOrMessage) {
  if (errorOrMessage === null || errorOrMessage === undefined) {
    return { rawName: '', rawMessage: '' };
  }
  if (typeof errorOrMessage === 'string') {
    return { rawName: '', rawMessage: errorOrMessage };
  }
  if (typeof errorOrMessage === 'object') {
    const rawName = typeof errorOrMessage.name === 'string' ? errorOrMessage.name : '';
    const rawMessage =
      typeof errorOrMessage.message === 'string'
        ? errorOrMessage.message
        : typeof errorOrMessage.reason === 'string'
          ? errorOrMessage.reason
          : String(errorOrMessage);
    return { rawName, rawMessage };
  }
  return { rawName: '', rawMessage: String(errorOrMessage) };
}

/**
 * Classifies common browser Wasm64 boot failures into a structured diagnostic
 * `{ code, category, hint, message }` with a sanitized message (zero host paths,
 * drive letters, usernames, or auth tokens).
 *
 * @param {unknown} errorOrMessage
 * @returns {{ code: string, category: string, hint: string, message: string }}
 */
export function classifyBootError(errorOrMessage) {
  const { rawName, rawMessage } = extractRawBootErrorText(errorOrMessage);
  const combined = rawName ? `${rawName}: ${rawMessage}` : rawMessage;
  const message = sanitizeString(rawMessage);

  if (WASM_MODULE_TOO_LARGE_RE.test(combined)) {
    return {
      code: BOOT_ERROR_CODES.WASM_MODULE_TOO_LARGE,
      category: 'wasm_compile',
      hint: 'Guest Wasm module exceeds V8 1 GiB compile limit (1073741824 bytes). Rebuild with --opt Oz --no-link-opt --strip to shrink the binary.',
      message,
    };
  }

  if (WASM64_UNSUPPORTED_RE.test(combined)) {
    return {
      code: BOOT_ERROR_CODES.WASM64_UNSUPPORTED,
      category: 'capability',
      hint: 'Browser lacks WebAssembly Memory64 (Wasm64) support. Use Chrome/Edge 133+ or Firefox 134+ with memory64 enabled.',
      message,
    };
  }

  if (COI_SAB_MISSING_RE.test(combined)) {
    return {
      code: BOOT_ERROR_CODES.COI_SAB_MISSING,
      category: 'capability',
      hint: 'SharedArrayBuffer requires cross-origin isolation. Serve with Cross-Origin-Opener-Policy: same-origin and Cross-Origin-Embedder-Policy: require-corp headers.',
      message,
    };
  }

  if (WORKER_SCRIPT_COLLISION_RE.test(combined)) {
    return {
      code: BOOT_ERROR_CODES.WORKER_SCRIPT_COLLISION,
      category: 'worker',
      hint: 'Classic worker importScripts identifier collision detected. Wrap helper scripts in UMD IIFEs so top-level const/let declarations do not collide.',
      message,
    };
  }

  if (HTTPFS_PGIM_RE.test(combined)) {
    return {
      code: BOOT_ERROR_CODES.HTTPFS_PGIM_ERROR,
      category: 'io',
      hint: 'HTTPFS/PGIM image load failed. Verify the local Range-capable HTTP server is running and /game/boot.pgim (or PGIM segments) are staged.',
      message,
    };
  }

  return {
    code: BOOT_ERROR_CODES.UNKNOWN_BOOT_ERROR,
    category: 'runtime',
    hint: 'Inspect the sanitized error message, stack trace, and bootDiagnostics snapshot for the failing boot stage.',
    message,
  };
}

function isPlainObject(value) {
  return Boolean(
    value &&
      typeof value === 'object' &&
      !Array.isArray(value) &&
      !isBinaryBuffer(value) &&
      !isSerializedNodeBuffer(value) &&
      !isBlobLike(value)
  );
}

const HEX_ADDRESS_RE = /^\s*(\+?)(?:0x)([0-9a-f]{1,16})\s*$/i;
const HEX_SHADER_HASH_RE = /^\s*(?:0x)?([0-9a-f]{1,64})\s*$/i;

function sanitizeHexAddress(value, { allowPlus = false, allowNumber = true } = {}) {
  if (value === null) {
    return null;
  }
  if (typeof value === 'bigint') {
    return value >= 0n ? `0x${value.toString(16)}` : undefined;
  }
  if (typeof value === 'number') {
    if (allowNumber && Number.isSafeInteger(value) && value >= 0) {
      return `0x${value.toString(16)}`;
    }
    return undefined;
  }
  if (typeof value === 'string') {
    const m = HEX_ADDRESS_RE.exec(value);
    if (m) {
      const prefix = allowPlus && m[1] === '+' ? '+' : '';
      return `${prefix}0x${m[2].toLowerCase()}`;
    }
  }
  return undefined;
}

function sanitizeInstructionOffset(value) {
  if (value === null) {
    return null;
  }
  if (typeof value === 'number' && Number.isSafeInteger(value) && value >= 0) {
    return value;
  }
  return sanitizeHexAddress(value, { allowPlus: true, allowNumber: false });
}

function roundTo2(value) {
  return Math.round(value * 100) / 100;
}

function roundTo3(value) {
  return Math.round(value * 1000) / 1000;
}

function clipSanitizedString(value, maxLen = 256) {
  if (typeof value !== 'string') {
    return undefined;
  }
  const safe = sanitizeString(value);
  return safe.length > maxLen ? `${safe.slice(0, maxLen - 3)}...` : safe;
}

function pickNonNegInt(obj, key) {
  const val = obj[key];
  if (typeof val === 'number' && Number.isFinite(val) && val >= 0) {
    return Math.floor(val);
  }
  return undefined;
}

function pickNonNegNumber(obj, key) {
  const val = obj[key];
  if (typeof val === 'number' && Number.isFinite(val) && val >= 0) {
    return val;
  }
  return undefined;
}

function pickPct(obj, key) {
  const val = obj[key];
  if (typeof val === 'number' && Number.isFinite(val)) {
    return Math.min(100, Math.max(0, val));
  }
  return undefined;
}

function pickUnitLevel(obj, key) {
  const val = obj[key];
  if (typeof val === 'number' && Number.isFinite(val)) {
    return Math.min(1, Math.max(0, val));
  }
  return undefined;
}

const ANSI_ESCAPE_RE = /\x1b\[[0-9;?]*[ -/]*[@-~]/g;
const GPU_ADAPTER_CTRL_CHAR_RE = /[\u0000-\u001f\u007f-\u009f]/g;
const GPU_ADAPTER_URL_RE = /\b(?:https?|file|data|javascript|ftp|wss?):\/\/[^\s"'`<>)\]},;]*/gi;
const GPU_ADAPTER_SCHEME_RE = /\b(?:javascript|data):[^\s"'`<>)\]},;]*/gi;
const GPU_ADAPTER_REDACTED_TOKEN_RE =
  /(?:\b(?:bearer|basic)\s+)?\??\[REDACTED(?:_[A-Z0-9_]+)?\]/gi;
const GPU_ADAPTER_LOWER_POSIX_PATH_RE =
  /(^|[\s"'`(\[={<,:;|])(\/(?:users|library|applications|volumes|home|root|tmp|var|etc|opt|usr|mnt|media|private|srv|proc|sys|dev|run|cygdrive)(?:\/[^\s"'`<>)\]},;]+)?)(?=[\s"'`<>)\]},;]|$)/gi;
const GPU_ADAPTER_MULTI_SEG_POSIX_PATH_RE =
  /(^|[\s"'`(\[=,:])\/(?:[A-Za-z0-9._-]+\/)+[A-Za-z0-9._~+-]*/g;
const GPU_ADAPTER_BACKSLASH_TOKEN_RE = /(?:^|\s)[^\s]*\\[^\s]*/g;
const GPU_ADAPTER_TRAVERSAL_RE = /\.\.(?:[\\/][^\s]*)?/g;

function sanitizeGpuAdapterText(rawText, limit) {
  if (typeof rawText !== 'string' || rawText.length === 0) {
    return null;
  }

  let text = rawText
    .replace(ANSI_ESCAPE_RE, ' ')
    .replace(/[\r\n\t]+/g, ' ')
    .replace(GPU_ADAPTER_CTRL_CHAR_RE, '');

  if (!text.trim()) {
    return null;
  }

  // Strip HTML/script fragments, URLs, query/fragment tokens, and host paths/secrets.
  text = text.replace(/<script\b[\s\S]*?<\/script>/gi, ' ');
  text = text.replace(/<[^>]*>/g, ' ');
  text = text.replace(GPU_ADAPTER_URL_RE, ' ');
  text = text.replace(GPU_ADAPTER_SCHEME_RE, ' ');
  text = sanitizeString(text);
  text = text.replace(GPU_ADAPTER_REDACTED_TOKEN_RE, ' ');
  text = text.replace(GPU_ADAPTER_BACKSLASH_TOKEN_RE, ' ');
  text = text.replace(GPU_ADAPTER_TRAVERSAL_RE, ' ');
  text = text.replace(GPU_ADAPTER_LOWER_POSIX_PATH_RE, '$1 ');
  text = text.replace(GPU_ADAPTER_MULTI_SEG_POSIX_PATH_RE, '$1 ');
  text = text.replace(/[?#][^\s]*/g, ' ');

  text = text
    .toLowerCase()
    .replace(/\(\s*\)|\[\s*\]/g, ' ')
    .replace(/[^a-z0-9 /._+()-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .replace(/\s*\/\s*/g, ' / ')
    .replace(/(?:\s*\/\s*){2,}/g, ' / ')
    .replace(/^[\s/._+-]+|[\s/._+-]+$/g, '');

  if (text.length > limit) {
    text = text.slice(0, limit).replace(/[\s/._+-]+$/, '');
  }

  if (!/[a-z0-9]/.test(text)) {
    return null;
  }

  return text.length > 0 ? text : null;
}

/**
 * Sanitizes a WebGPU adapter label (or `{ vendor, architecture }` / `GPUAdapterInfo` object)
 * into a lowercase safe vendor/architecture string of at most `MAX_GPU_ADAPTER_LENGTH` (96)
 * characters with zero host paths, URLs, control characters, or secrets.
 *
 * @param {unknown} rawAdapter
 * @param {number} [maxLength=MAX_GPU_ADAPTER_LENGTH]
 * @returns {string|null}
 */
export function sanitizeGpuAdapter(rawAdapter, maxLength = MAX_GPU_ADAPTER_LENGTH) {
  if (rawAdapter === null || rawAdapter === undefined) {
    return null;
  }

  const limit =
    typeof maxLength === 'number' && Number.isFinite(maxLength) && maxLength >= 1
      ? Math.min(MAX_GPU_ADAPTER_LENGTH, Math.floor(maxLength))
      : MAX_GPU_ADAPTER_LENGTH;

  if (typeof rawAdapter === 'string') {
    return sanitizeGpuAdapterText(rawAdapter, limit);
  }

  if (!isPlainObject(rawAdapter)) {
    return null;
  }

  const source = isPlainObject(rawAdapter.info) ? rawAdapter.info : rawAdapter;
  const rawVendor = typeof source.vendor === 'string' ? source.vendor : null;
  const rawArch =
    typeof source.architecture === 'string'
      ? source.architecture
      : typeof source.arch === 'string'
        ? source.arch
        : null;

  if (rawVendor !== null || rawArch !== null) {
    const safeVendor = rawVendor !== null ? sanitizeGpuAdapterText(rawVendor, limit) : null;
    const safeArch = rawArch !== null ? sanitizeGpuAdapterText(rawArch, limit) : null;
    if (safeVendor && safeArch) {
      if (safeVendor === safeArch) {
        return safeVendor;
      }
      return sanitizeGpuAdapterText(`${safeVendor} / ${safeArch}`, limit);
    }
    if (safeVendor || safeArch) {
      return safeVendor || safeArch;
    }
  }

  for (const key of ['gpuAdapter', 'gpu_adapter', 'adapterLabel', 'adapter_label', 'adapter', 'label', 'description', 'device', 'name']) {
    if (typeof source[key] === 'string' && source[key].length > 0) {
      const cleaned = sanitizeGpuAdapterText(source[key], limit);
      if (cleaned !== null) {
        return cleaned;
      }
    }
  }

  return null;
}

const GUEST_AUDIO_RING_KEYS = Object.freeze([
  'guestRingAttached',
  'guestFramesDrained',
  'guestUnderruns',
  'guestOverruns',
  'guestPeakLevel',
]);

function hasGuestAudioRingFields(obj) {
  if (!isPlainObject(obj)) {
    return false;
  }
  for (let i = 0; i < GUEST_AUDIO_RING_KEYS.length; i += 1) {
    if (obj[GUEST_AUDIO_RING_KEYS[i]] !== undefined) {
      return true;
    }
  }
  return false;
}

/**
 * Strictly allowlists and sanitizes `#1027` guest audio PCM ring telemetry
 * (and safe AudioEngine/worker summary counters, supporting duck-typed
 * `AudioEngine` instances via `getStats()` / `getTelemetry()`).
 * Drops raw PCM buffers, host paths, secrets, and unknown keys.
 *
 * @param {unknown} raw
 * @returns {object|null}
 */
function sanitizeAudioTelemetry(raw) {
  if (raw === null || raw === undefined) {
    return null;
  }
  let source = raw;
  if (typeof source === 'object' && typeof source.getStats === 'function') {
    try {
      source = source.getStats();
    } catch {
      return null;
    }
  } else if (typeof source === 'object' && typeof source.getTelemetry === 'function') {
    try {
      source = source.getTelemetry();
    } catch {
      return null;
    }
  }
  if (!isPlainObject(source)) {
    return null;
  }

  const nestedGuest = isPlainObject(source.guestAudio)
    ? source.guestAudio
    : isPlainObject(source.guestRing)
      ? source.guestRing
      : null;

  const out = {};

  // #1027 guest audio PCM ring counters (strictly typed & bounded)
  const rawAttached =
    source.guestRingAttached !== undefined
      ? source.guestRingAttached
      : nestedGuest
        ? nestedGuest.guestRingAttached ?? nestedGuest.attached
        : undefined;
  if (typeof rawAttached === 'boolean') {
    out.guestRingAttached = rawAttached;
  }

  for (const intKey of ['guestFramesDrained', 'guestUnderruns', 'guestOverruns']) {
    let val = pickNonNegInt(source, intKey);
    if (val === undefined && nestedGuest) {
      val = pickNonNegInt(nestedGuest, intKey);
    }
    if (val !== undefined) {
      out[intKey] = val;
    }
  }

  let guestPeakLevel = pickUnitLevel(source, 'guestPeakLevel');
  if (guestPeakLevel === undefined && nestedGuest) {
    guestPeakLevel = pickUnitLevel(nestedGuest, 'guestPeakLevel');
  }
  if (guestPeakLevel !== undefined) {
    out.guestPeakLevel = guestPeakLevel;
  }

  // Safe AudioEngine / worker status metadata
  for (const strKey of ['outputState', 'contextState', 'state']) {
    const val = clipSanitizedString(source[strKey], 32);
    if (val !== undefined) {
      out[strKey] = val;
    }
  }

  for (const intKey of [
    'sampleRate',
    'channels',
    'capacityFrames',
    'bufferedFrames',
    'freeFrames',
    'submittedFrames',
    'consumedFrames',
    'underruns',
    'underrunFrames',
    'overflows',
    'overruns',
    'droppedFrames',
    'activeVoices',
  ]) {
    const val = pickNonNegInt(source, intKey);
    if (val !== undefined) {
      out[intKey] = val;
    }
  }

  for (const levelKey of ['gain', 'volume', 'masterGain', 'effectiveGain']) {
    const val = pickUnitLevel(source, levelKey);
    if (val !== undefined) {
      out[levelKey] = val;
    }
  }

  for (const boolKey of ['muted', 'unlockedOnGesture']) {
    if (typeof source[boolKey] === 'boolean') {
      out[boolKey] = source[boolKey];
    }
  }

  return out;
}

function extractMilestoneElapsedMs(milestones, name) {
  if (!Array.isArray(milestones)) {
    return null;
  }
  for (let i = 0; i < milestones.length; i += 1) {
    const m = milestones[i];
    if (
      isPlainObject(m) &&
      m.milestone === name &&
      m.reached !== false &&
      typeof m.elapsedMs === 'number' &&
      Number.isFinite(m.elapsedMs) &&
      m.elapsedMs >= 0
    ) {
      return m.elapsedMs;
    }
  }
  return null;
}

function resolveBootTimingMetrics({
  explicitFirstVisibleFrameMs,
  explicitBlackLoadingMs,
  bootSnapshot,
  gpuTelemetry,
}) {
  const gpuFs = gpuTelemetry && isPlainObject(gpuTelemetry.frameStats) ? gpuTelemetry.frameStats : null;

  const firstPresentMs =
    (bootSnapshot && pickNonNegNumber(bootSnapshot, 'firstPresentMs')) ??
    (gpuTelemetry && pickNonNegNumber(gpuTelemetry, 'firstPresentMs')) ??
    (gpuFs && pickNonNegNumber(gpuFs, 'firstPresentMs')) ??
    (bootSnapshot ? extractMilestoneElapsedMs(bootSnapshot.milestones, 'FIRST_PRESENT') : null) ??
    null;

  let firstVisibleFrameMs = null;
  if (explicitFirstVisibleFrameMs === null) {
    firstVisibleFrameMs = null;
  } else if (
    typeof explicitFirstVisibleFrameMs === 'number' &&
    Number.isFinite(explicitFirstVisibleFrameMs) &&
    explicitFirstVisibleFrameMs >= 0
  ) {
    firstVisibleFrameMs = explicitFirstVisibleFrameMs;
  } else {
    firstVisibleFrameMs =
      (bootSnapshot && pickNonNegNumber(bootSnapshot, 'firstVisibleFrameMs')) ??
      (gpuTelemetry && pickNonNegNumber(gpuTelemetry, 'firstVisibleFrameMs')) ??
      (gpuFs && pickNonNegNumber(gpuFs, 'firstVisibleFrameMs')) ??
      (bootSnapshot ? extractMilestoneElapsedMs(bootSnapshot.milestones, 'FIRST_VISIBLE_FRAME') : null) ??
      null;
  }

  let blackLoadingMs = null;
  if (explicitBlackLoadingMs === null) {
    blackLoadingMs = null;
  } else if (
    typeof explicitBlackLoadingMs === 'number' &&
    Number.isFinite(explicitBlackLoadingMs) &&
    explicitBlackLoadingMs >= 0
  ) {
    blackLoadingMs = explicitBlackLoadingMs;
  } else {
    const explicitFromSources =
      (bootSnapshot ? pickNonNegNumber(bootSnapshot, 'blackLoadingMs') : undefined) ??
      (gpuTelemetry ? pickNonNegNumber(gpuTelemetry, 'blackLoadingMs') : undefined) ??
      (gpuFs ? pickNonNegNumber(gpuFs, 'blackLoadingMs') : undefined);
    if (explicitFromSources !== undefined) {
      blackLoadingMs = explicitFromSources;
    } else if (firstVisibleFrameMs !== null && firstPresentMs !== null) {
      blackLoadingMs = Math.max(0, firstVisibleFrameMs - firstPresentMs);
    }
  }

  return { firstVisibleFrameMs, blackLoadingMs };
}

function sanitizeTrapRecord(raw) {
  if (!isPlainObject(raw)) {
    return null;
  }
  const out = {};
  const seq = pickNonNegInt(raw, 'seq');
  if (seq !== undefined) out.seq = seq;
  const kind = clipSanitizedString(raw.kind, 64);
  if (kind !== undefined) out.kind = kind;
  if (raw.guestVa === null) {
    out.guestVa = null;
  } else if (raw.guestVa !== undefined) {
    const hexVa = sanitizeHexAddress(raw.guestVa);
    if (hexVa !== undefined) {
      out.guestVa = hexVa;
    } else {
      const guestVa = clipSanitizedString(raw.guestVa, 32);
      if (guestVa !== undefined) out.guestVa = guestVa;
    }
  }
  for (const offsetKey of ['instructionOffset', 'insnOffset', 'rva', 'funcOffset']) {
    if (raw[offsetKey] !== undefined) {
      const cleanOffset = sanitizeInstructionOffset(raw[offsetKey]);
      if (cleanOffset !== undefined) out[offsetKey] = cleanOffset;
    }
  }
  for (const addrKey of ['functionVa', 'funcVa', 'rip', 'pc']) {
    if (raw[addrKey] !== undefined) {
      const cleanAddr = sanitizeHexAddress(raw[addrKey]);
      if (cleanAddr !== undefined) out[addrKey] = cleanAddr;
    }
  }
  if (raw.opcode === null) {
    out.opcode = null;
  } else if (typeof raw.opcode === 'number' && Number.isSafeInteger(raw.opcode) && raw.opcode >= 0) {
    out.opcode = raw.opcode;
  } else if (typeof raw.opcode === 'string') {
    const opcode = clipSanitizedString(raw.opcode, 64);
    if (opcode !== undefined) out.opcode = opcode;
  }
  if (raw.mnemonic === null) {
    out.mnemonic = null;
  } else {
    const mnemonic = clipSanitizedString(raw.mnemonic, 64);
    if (mnemonic !== undefined) out.mnemonic = mnemonic;
  }
  if (raw.dll === null) {
    out.dll = null;
  } else {
    const dll = clipSanitizedString(raw.dll, 128);
    if (dll !== undefined) out.dll = dll;
  }
  if (raw.symbol === null) {
    out.symbol = null;
  } else {
    const symbol = clipSanitizedString(raw.symbol, 256);
    if (symbol !== undefined) out.symbol = symbol;
  }
  if (raw.import === null) {
    out.import = null;
  } else {
    const imp = clipSanitizedString(raw.import, 256);
    if (imp !== undefined) out.import = imp;
  }
  const message = clipSanitizedString(raw.message, 512);
  if (message !== undefined) out.message = message;
  if (typeof raw.fatal === 'boolean') out.fatal = raw.fatal;
  const elapsedMs = pickNonNegNumber(raw, 'elapsedMs');
  if (elapsedMs !== undefined) out.elapsedMs = elapsedMs;
  const stage = clipSanitizedString(raw.stage, 64);
  if (stage !== undefined) out.stage = stage;
  const count = pickNonNegInt(raw, 'count');
  if (count !== undefined) out.count = count;
  return out;
}

function sanitizeUnresolvedImportRecord(raw) {
  if (!isPlainObject(raw)) {
    return null;
  }
  const out = {};
  const name = clipSanitizedString(raw.name, 256);
  if (name !== undefined) out.name = name;
  if (raw.dll === null) {
    out.dll = null;
  } else {
    const dll = clipSanitizedString(raw.dll, 128);
    if (dll !== undefined) out.dll = dll;
  }
  if (raw.symbol === null) {
    out.symbol = null;
  } else {
    const symbol = clipSanitizedString(raw.symbol, 256);
    if (symbol !== undefined) out.symbol = symbol;
  }
  if (raw.ordinal === null) {
    out.ordinal = null;
  } else {
    const ordinal = pickNonNegInt(raw, 'ordinal');
    if (ordinal !== undefined) out.ordinal = ordinal;
  }
  if (raw.guestVa === null) {
    out.guestVa = null;
  } else if (raw.guestVa !== undefined) {
    const hexVa = sanitizeHexAddress(raw.guestVa);
    if (hexVa !== undefined) {
      out.guestVa = hexVa;
    } else {
      const guestVa = clipSanitizedString(raw.guestVa, 32);
      if (guestVa !== undefined) out.guestVa = guestVa;
    }
  }
  for (const offsetKey of ['instructionOffset', 'insnOffset', 'rva']) {
    if (raw[offsetKey] !== undefined) {
      const cleanOffset = sanitizeInstructionOffset(raw[offsetKey]);
      if (cleanOffset !== undefined) out[offsetKey] = cleanOffset;
    }
  }
  const count = pickNonNegInt(raw, 'count');
  if (count !== undefined) out.count = count;
  const firstElapsedMs = pickNonNegNumber(raw, 'firstElapsedMs');
  if (firstElapsedMs !== undefined) out.firstElapsedMs = firstElapsedMs;
  const lastElapsedMs = pickNonNegNumber(raw, 'lastElapsedMs');
  if (lastElapsedMs !== undefined) out.lastElapsedMs = lastElapsedMs;
  return out;
}

function sanitizeMilestoneRecord(raw) {
  if (!isPlainObject(raw)) {
    return null;
  }
  const out = {};
  const milestone = clipSanitizedString(raw.milestone, 64);
  if (milestone !== undefined) out.milestone = milestone;
  const index = pickNonNegInt(raw, 'index');
  if (index !== undefined) out.index = index;
  if (typeof raw.reached === 'boolean') out.reached = raw.reached;
  if (raw.elapsedMs === null) {
    out.elapsedMs = null;
  } else {
    const elapsedMs = pickNonNegNumber(raw, 'elapsedMs');
    if (elapsedMs !== undefined) out.elapsedMs = elapsedMs;
  }
  if (raw.lastElapsedMs === null) {
    out.lastElapsedMs = null;
  } else {
    const lastElapsedMs = pickNonNegNumber(raw, 'lastElapsedMs');
    if (lastElapsedMs !== undefined) out.lastElapsedMs = lastElapsedMs;
  }
  const count = pickNonNegInt(raw, 'count');
  if (count !== undefined) out.count = count;
  if (raw.detail === null) {
    out.detail = null;
  } else if (isPlainObject(raw.detail)) {
    const detail = {};
    const source = clipSanitizedString(raw.detail.source, 64);
    if (source !== undefined) detail.source = source;
    const line = clipSanitizedString(raw.detail.line, 256);
    if (line !== undefined) detail.line = line;
    if (Object.keys(detail).length > 0) {
      out.detail = detail;
    }
  }
  return out;
}

/**
 * Strictly allowlists and sanitizes a boot summary snapshot (or BootTracer instance).
 * Drops unknown keys, binary buffers, and redacts any host paths or secrets in strings.
 *
 * @param {unknown} raw
 * @returns {object|null}
 */
function sanitizeBootSnapshot(raw) {
  if (raw === null || raw === undefined) {
    return null;
  }
  let source = raw;
  if (typeof source === 'object' && typeof source.getSummary === 'function') {
    try {
      source = source.getSummary();
    } catch {
      return null;
    }
  }
  if (!isPlainObject(source)) {
    return null;
  }

  const out = {};

  const stage = clipSanitizedString(source.stage, 64);
  if (stage !== undefined) out.stage = stage;
  const bootPhase = clipSanitizedString(source.bootPhase, 64);
  if (bootPhase !== undefined) out.bootPhase = bootPhase;
  const phase = clipSanitizedString(source.phase, 64);
  if (phase !== undefined) out.phase = phase;
  const mode = clipSanitizedString(source.mode, 64);
  if (mode !== undefined) out.mode = mode;
  const bootMode = clipSanitizedString(source.bootMode, 64);
  if (bootMode !== undefined) out.bootMode = bootMode;
  const status = clipSanitizedString(source.status, 64);
  if (status !== undefined) out.status = status;

  const progressPct = pickPct(source, 'progressPct');
  if (progressPct !== undefined) out.progressPct = progressPct;
  if (typeof source.readyForSplash === 'boolean') out.readyForSplash = source.readyForSplash;

  const elapsedMs = pickNonNegNumber(source, 'elapsedMs');
  if (elapsedMs !== undefined) out.elapsedMs = elapsedMs;
  const phaseElapsedMs = pickNonNegNumber(source, 'phaseElapsedMs');
  if (phaseElapsedMs !== undefined) out.phaseElapsedMs = phaseElapsedMs;
  const presentFpsAvg = pickNonNegNumber(source, 'presentFpsAvg');
  if (presentFpsAvg !== undefined) out.presentFpsAvg = presentFpsAvg;
  const budgetCompliancePct = pickPct(source, 'budgetCompliancePct');
  if (budgetCompliancePct !== undefined) out.budgetCompliancePct = budgetCompliancePct;
  const targetFps = pickNonNegNumber(source, 'targetFps');
  if (targetFps !== undefined) out.targetFps = targetFps;
  const budgetMs = pickNonNegNumber(source, 'budgetMs') ?? pickNonNegNumber(source, 'frameBudgetMs');
  if (budgetMs !== undefined) out.budgetMs = budgetMs;
  const dominantBottleneck = clipSanitizedString(source.dominantBottleneck, 64);
  if (dominantBottleneck !== undefined) out.dominantBottleneck = dominantBottleneck;
  if (source.firstPresentMs === null) {
    out.firstPresentMs = null;
  } else {
    const firstPresentMs = pickNonNegNumber(source, 'firstPresentMs');
    if (firstPresentMs !== undefined) out.firstPresentMs = firstPresentMs;
  }
  if (source.firstVisibleFrameMs === null) {
    out.firstVisibleFrameMs = null;
  } else {
    const firstVisibleFrameMs = pickNonNegNumber(source, 'firstVisibleFrameMs');
    if (firstVisibleFrameMs !== undefined) out.firstVisibleFrameMs = firstVisibleFrameMs;
  }
  if (source.blackLoadingMs === null) {
    out.blackLoadingMs = null;
  } else {
    const blackLoadingMs = pickNonNegNumber(source, 'blackLoadingMs');
    if (blackLoadingMs !== undefined) out.blackLoadingMs = blackLoadingMs;
  }
  if (source.gpuAdapter === null) {
    out.gpuAdapter = null;
  } else if (source.gpuAdapter !== undefined) {
    const gpuAdapter = sanitizeGpuAdapter(source.gpuAdapter);
    if (gpuAdapter !== null) out.gpuAdapter = gpuAdapter;
  }
  const milestonesReached = pickNonNegInt(source, 'milestonesReached');
  if (milestonesReached !== undefined) out.milestonesReached = milestonesReached;

  if (Array.isArray(source.milestones)) {
    const milestones = [];
    for (let i = 0; i < source.milestones.length && milestones.length < MAX_BOOT_MILESTONES; i += 1) {
      const clean = sanitizeMilestoneRecord(source.milestones[i]);
      if (clean && Object.keys(clean).length > 0) {
        milestones.push(clean);
      }
    }
    out.milestones = milestones;
  }

  const trapCount = pickNonNegInt(source, 'trapCount');
  if (trapCount !== undefined) out.trapCount = trapCount;
  const uniqueTrapCount = pickNonNegInt(source, 'uniqueTrapCount');
  if (uniqueTrapCount !== undefined) out.uniqueTrapCount = uniqueTrapCount;

  if (source.lastTrap === null) {
    out.lastTrap = null;
  } else if (source.lastTrap !== undefined) {
    const cleanTrap = sanitizeTrapRecord(source.lastTrap);
    if (cleanTrap) {
      out.lastTrap = cleanTrap;
    }
  }

  if (Array.isArray(source.unresolvedImports)) {
    const unresolvedImports = [];
    for (
      let i = 0;
      i < source.unresolvedImports.length && unresolvedImports.length < MAX_BOOT_UNRESOLVED_IMPORTS;
      i += 1
    ) {
      const clean = sanitizeUnresolvedImportRecord(source.unresolvedImports[i]);
      if (clean && Object.keys(clean).length > 0) {
        unresolvedImports.push(clean);
      }
    }
    out.unresolvedImports = unresolvedImports;
  }

  if (Array.isArray(source.recentTraps)) {
    const recentTraps = [];
    for (let i = 0; i < source.recentTraps.length && recentTraps.length < MAX_BOOT_TRAPS; i += 1) {
      const clean = sanitizeTrapRecord(source.recentTraps[i]);
      if (clean && Object.keys(clean).length > 0) {
        recentTraps.push(clean);
      }
    }
    out.recentTraps = recentTraps;
  }

  const wasmModuleBytes = pickNonNegNumber(source, 'wasmModuleBytes');
  if (wasmModuleBytes !== undefined) out.wasmModuleBytes = wasmModuleBytes;
  const wasmSizeBytes = pickNonNegNumber(source, 'wasmSizeBytes');
  if (wasmSizeBytes !== undefined) out.wasmSizeBytes = wasmSizeBytes;
  const pgimBytes = pickNonNegNumber(source, 'pgimBytes');
  if (pgimBytes !== undefined) out.pgimBytes = pgimBytes;
  const pgimSegments = pickNonNegInt(source, 'pgimSegments');
  if (pgimSegments !== undefined) out.pgimSegments = pgimSegments;
  const shaderPackCount = pickNonNegInt(source, 'shaderPackCount');
  if (shaderPackCount !== undefined) out.shaderPackCount = shaderPackCount;

  if (typeof source.updatedAt === 'string') {
    const updatedAt = clipSanitizedString(source.updatedAt, 64);
    if (updatedAt !== undefined) out.updatedAt = updatedAt;
  } else {
    const updatedAtNum = pickNonNegNumber(source, 'updatedAt');
    if (updatedAtNum !== undefined) out.updatedAt = updatedAtNum;
  }

  return out;
}

function sanitizeMissingShaderEntry(raw) {
  if (!isPlainObject(raw)) {
    return null;
  }
  let hash;
  if (typeof raw.hash === 'bigint' && raw.hash >= 0n) {
    hash = `0x${raw.hash.toString(16).padStart(16, '0')}`;
  } else if (typeof raw.hash === 'number' && Number.isSafeInteger(raw.hash) && raw.hash >= 0) {
    hash = `0x${raw.hash.toString(16).padStart(16, '0')}`;
  } else if (typeof raw.hash === 'string') {
    const hexMatch = HEX_SHADER_HASH_RE.exec(raw.hash);
    hash = hexMatch ? `0x${hexMatch[1].toLowerCase()}` : clipSanitizedString(raw.hash, 64);
  } else if (typeof raw.hashHex === 'string') {
    const hexMatch = HEX_SHADER_HASH_RE.exec(raw.hashHex);
    hash = hexMatch ? `0x${hexMatch[1].toLowerCase()}` : clipSanitizedString(raw.hashHex, 64);
  }
  const stage = clipSanitizedString(raw.stage, 32);
  const hits = pickNonNegInt(raw, 'hits') ?? pickNonNegInt(raw, 'count');
  const firstSeenFrame = pickNonNegInt(raw, 'firstSeenFrame');
  const pipelineKind = clipSanitizedString(raw.pipelineKind ?? raw.profile, 64);

  if (hash === undefined && stage === undefined && hits === undefined) {
    return null;
  }
  const out = {};
  if (hash !== undefined) out.hash = hash;
  if (stage !== undefined) out.stage = stage;
  if (hits !== undefined) out.hits = hits;
  if (firstSeenFrame !== undefined) out.firstSeenFrame = firstSeenFrame;
  if (pipelineKind !== undefined) out.pipelineKind = pipelineKind;
  return out;
}

/**
 * Strictly allowlists and sanitizes a Bink2 / splash video telemetry snapshot
 * (or duck-typed `BinkVideoPresenter` / `SplashFrameBridge` instance via `getStats()`).
 * Drops raw pixel planes/buffers, file/video paths, URLs, and unknown keys.
 *
 * @param {unknown} raw
 * @param {number} [depth=0]
 * @returns {object|null}
 */
function sanitizeBinkTelemetry(raw, depth = 0) {
  if (raw === null || raw === undefined) {
    return null;
  }
  let source = raw;
  if (typeof source === 'object' && typeof source.getStats === 'function') {
    try {
      source = source.getStats();
    } catch {
      return null;
    }
  }
  if (!isPlainObject(source)) {
    return null;
  }

  const out = {};
  for (const strKey of [
    'pixelFormat',
    'lastFrameFormat',
    'colorSpace',
    'colorRange',
    'chromaOrder',
    'lastChromaOrder',
    'loadOp',
    'lastLoadOp',
    'presentationFormat',
    'lastSource',
    'state',
    'status',
  ]) {
    const val = clipSanitizedString(source[strKey], 64);
    if (val !== undefined) out[strKey] = val;
  }

  for (const intKey of [
    'frameWidth',
    'frameHeight',
    'lastFrameWidth',
    'lastFrameHeight',
    'uploadedFrames',
    'presentedFrames',
    'framesPresented',
    'droppedFrames',
    'skippedFrames',
    'decodedFrames',
    'queueDepth',
    'maxQueueDepth',
    'resizeCount',
    'planeReallocations',
    'scratchAllocations',
    'directZeroCopyUploads',
    'chromaRepackUploads',
    'bytesUploaded',
    'slotCount',
    'activeSlotIndex',
    'pendingUploads',
    'framesBeforeFirstPresent',
    'lateSplashFramesIgnored',
    'nv12FramesForwarded',
    'yvuFramesForwarded',
    'underlayLoadFramesPresented',
    'zeroCopyFramesForwarded',
    'audioTracks',
    'sampleRate',
    'channels',
    'audioUnderruns',
  ]) {
    const val = pickNonNegInt(source, intKey);
    if (val !== undefined) out[intKey] = val;
  }

  for (const numKey of ['targetFps', 'frameIntervalMs', 'avgDecodeMs', 'avgUploadMs', 'fps', 'onePercentLowFps']) {
    const val = pickNonNegNumber(source, numKey);
    if (val !== undefined) out[numKey] = val;
  }

  for (const boolKey of [
    'hasFrame',
    'paused',
    'looping',
    'skipped',
    'pipelineReady',
    'usesStagingPool',
    'firstPresentSent',
    'guestFirstPresent',
    'splashSuppressed',
    'zeroCopy',
    'destroyed',
    'active',
  ]) {
    if (typeof source[boolKey] === 'boolean') {
      out[boolKey] = source[boolKey];
    }
  }

  if (depth < 1 && isPlainObject(source.presenterStats)) {
    const nested = sanitizeBinkTelemetry(source.presenterStats, depth + 1);
    if (nested && Object.keys(nested).length > 0) {
      out.presenterStats = nested;
    }
  }

  return out;
}

function sanitizeIoWorkerStatsSnapshot(raw) {
  if (raw === null || raw === undefined) {
    return null;
  }
  let source = raw;
  if (typeof source === 'object' && typeof source.getStats === 'function') {
    try {
      source = source.getStats();
    } catch {
      return null;
    }
  }
  if (!isPlainObject(source)) {
    return null;
  }
  const mailbox = isPlainObject(source.mailbox) ? source.mailbox : null;
  const streaming = isPlainObject(source.streaming) ? source.streaming : null;
  const imm = isPlainObject(source.immutableCache) ? source.immutableCache : null;
  const out = {};

  if (typeof source.mailboxAttached === 'boolean') {
    out.mailboxAttached = source.mailboxAttached;
  }
  const intSpecs = [
    ['mailboxLanes', source.mailboxLanes],
    ['openHandles', source.openHandles],
    ['readsCompleted', source.readsCompleted ?? (streaming ? streaming.readsCompleted : undefined)],
    ['bytesTransferred', source.bytesTransferred ?? (streaming ? streaming.bytesTransferred : undefined)],
    [
      'inFlightRequests',
      source.inFlightRequests ??
        (mailbox ? mailbox.activeLanes ?? mailbox.activePendingSlots : undefined) ??
        (imm ? imm.inFlightRequests : undefined),
    ],
    ['activeLanes', source.activeLanes ?? (mailbox ? mailbox.activeLanes : undefined)],
    ['peakActiveLanes', source.peakActiveLanes ?? (mailbox ? mailbox.peakActiveLanes : undefined)],
    [
      'requests',
      source.requests ??
        (mailbox ? mailbox.totalRequests ?? mailbox.requests : undefined),
    ],
    ['syncFastPathHits', source.syncFastPathHits ?? (mailbox ? mailbox.syncFastPathHits : undefined)],
    ['asyncPathHits', source.asyncPathHits ?? (mailbox ? mailbox.asyncPathHits : undefined)],
    ['errorCount', source.errorCount ?? (streaming ? streaming.errorCount : undefined)],
    ['cacheHits', source.cacheHits ?? (streaming ? streaming.cacheHits : undefined)],
    ['cacheMisses', source.cacheMisses],
    ['bytesServedFromCache', source.bytesServedFromCache],
    ['bytesFetched', source.bytesFetched],
    ['slabCacheHits', source.slabCacheHits ?? (imm ? imm.slabCacheHits : undefined)],
    ['slabCacheMisses', source.slabCacheMisses ?? (imm ? imm.slabCacheMisses : undefined)],
    ['opfsHits', source.opfsHits],
    ['opfsMisses', source.opfsMisses],
    ['statHits', source.statHits],
    ['statMisses', source.statMisses],
  ];
  for (let i = 0; i < intSpecs.length; i += 1) {
    const [key, val] = intSpecs[i];
    if (typeof val === 'number' && Number.isFinite(val) && val >= 0) {
      out[key] = Math.floor(val);
    }
  }
  const avgReadMs = source.avgReadMs ?? (streaming ? streaming.avgReadMs : undefined);
  if (typeof avgReadMs === 'number' && Number.isFinite(avgReadMs) && avgReadMs >= 0) {
    out.avgReadMs = Math.round(avgReadMs * 1000) / 1000;
  }

  return Object.keys(out).length > 0 ? out : null;
}

function sanitizeRuntimeDiagnosticsSnapshot(raw) {
  if (!isPlainObject(raw)) {
    return null;
  }
  const out = {};
  if (isPlainObject(raw.dstorage)) {
    const ds = {};
    for (const k of [
      'requests_completed',
      'requests_failed',
      'requests_cancelled',
      'bytes_delivered',
      'file_reads',
      'coalesced_requests',
      'decompressed_requests',
      'submits',
      'live_queues',
      'physical_page_requests',
      'bcpack_requests',
      'bcpack_decoded',
    ]) {
      const v = pickNonNegInt(raw.dstorage, k);
      if (v !== undefined) ds[k] = v;
    }
    const dbg = clipSanitizedString(raw.dstorage.debug_state, 512);
    if (dbg !== undefined) ds.debug_state = dbg;
    if (Object.keys(ds).length > 0) out.dstorage = ds;
  }
  if (raw.blocked_waits !== undefined && raw.blocked_waits !== null) {
    out.blocked_waits = sanitizeValue(raw.blocked_waits);
  }
  if (isPlainObject(raw.streaming_drain)) {
    out.streaming_drain = sanitizeValue(raw.streaming_drain);
  }
  return Object.keys(out).length > 0 ? out : null;
}

/**
 * Strictly allowlists and sanitizes a GPU telemetry snapshot.
 * Drops unknown keys, shader sources, and raw binary buffers; redacts host paths/tokens;
 * preserves exact validation/draw/skip counts, 1%-low FPS, Bink2 telemetry, and missing shader hashes.
 *
 * @param {unknown} raw
 * @returns {object|null}
 */
function sanitizeGpuTelemetry(raw) {
  if (raw === null || raw === undefined) {
    return null;
  }
  if (!isPlainObject(raw)) {
    return null;
  }

  const out = {};

  const timestamp = pickNonNegNumber(raw, 'timestamp');
  if (timestamp !== undefined) out.timestamp = timestamp;
  const adapter = clipSanitizedString(raw.adapter, 128);
  if (adapter !== undefined) out.adapter = adapter;
  if (raw.gpuAdapter === null) {
    out.gpuAdapter = null;
  } else {
    const rawGpuAdapter =
      raw.gpuAdapter !== undefined
        ? raw.gpuAdapter
        : raw.adapterLabel !== undefined
          ? raw.adapterLabel
          : raw.adapterInfo !== undefined
            ? raw.adapterInfo
            : raw.adapter;
    if (rawGpuAdapter !== undefined) {
      const gpuAdapter = sanitizeGpuAdapter(rawGpuAdapter);
      if (gpuAdapter !== null) out.gpuAdapter = gpuAdapter;
    }
  }
  const backend = clipSanitizedString(raw.backend, 64);
  if (backend !== undefined) out.backend = backend;
  if (typeof raw.deviceLost === 'boolean') out.deviceLost = raw.deviceLost;
  if (raw.firstPresentMs === null) {
    out.firstPresentMs = null;
  } else {
    const firstPresentMs = pickNonNegNumber(raw, 'firstPresentMs');
    if (firstPresentMs !== undefined) out.firstPresentMs = firstPresentMs;
  }
  if (raw.firstVisibleFrameMs === null) {
    out.firstVisibleFrameMs = null;
  } else {
    const firstVisibleFrameMs = pickNonNegNumber(raw, 'firstVisibleFrameMs');
    if (firstVisibleFrameMs !== undefined) out.firstVisibleFrameMs = firstVisibleFrameMs;
  }
  if (raw.blackLoadingMs === null) {
    out.blackLoadingMs = null;
  } else {
    const blackLoadingMs = pickNonNegNumber(raw, 'blackLoadingMs');
    if (blackLoadingMs !== undefined) out.blackLoadingMs = blackLoadingMs;
  }

  if (raw.lastValidationErrorCode === null) {
    out.lastValidationErrorCode = null;
  } else {
    const code = clipSanitizedString(raw.lastValidationErrorCode, 128);
    if (code !== undefined) out.lastValidationErrorCode = code;
  }
  if (raw.lastValidationErrorMessage === null) {
    out.lastValidationErrorMessage = null;
  } else {
    const msg = clipSanitizedString(raw.lastValidationErrorMessage, 512);
    if (msg !== undefined) out.lastValidationErrorMessage = msg;
  }

  for (const intKey of [
    'validationErrors',
    'validationErrorCount',
    'validationErrorSuppressedCount',
    'shaderPackCount',
    'loadedShaders',
    'pipelineCount',
    'bufferCount',
    'textureCount',
    'framesPresented',
    'presentCount',
    'splashFramesPresented',
    'frameCount',
    'droppedFrames',
    'drawCalls',
    'realPackDrawCount',
    'fallbackDrawCount',
    'skippedDrawCount',
    'skippedDraws',
    'unboundBindGroupCount',
    'computeDispatches',
    'packetsProcessed',
    'packetCount',
    'bytesProcessed',
    'vramBytes',
    'framesWithin30FpsBudget',
    'framesOver30FpsBudget',
  ]) {
    const val = pickNonNegInt(raw, intKey);
    if (val !== undefined) out[intKey] = val;
  }

  for (const numKey of [
    'fps',
    'presentFpsAvg',
    'onePercentLowFps',
    'fps1PctLow',
    'p95FrameMs',
    'p99FrameMs',
    'maxFrameMs',
    'targetFps',
    'budgetMs',
    'frameBudgetMs',
  ]) {
    const val = pickNonNegNumber(raw, numKey);
    if (val !== undefined) out[numKey] = val;
  }

  const gpuBudgetCompliancePct = pickPct(raw, 'budgetCompliancePct');
  if (gpuBudgetCompliancePct !== undefined) out.budgetCompliancePct = gpuBudgetCompliancePct;
  const gpuComplianceRate = pickUnitLevel(raw, 'fps30ComplianceRate');
  if (gpuComplianceRate !== undefined) out.fps30ComplianceRate = gpuComplianceRate;
  const gpuDominantBottleneck = clipSanitizedString(raw.dominantBottleneck, 64);
  if (gpuDominantBottleneck !== undefined) out.dominantBottleneck = gpuDominantBottleneck;

  if (isPlainObject(raw.frameStats)) {
    const fs = {};
    for (const intKey of [
      'framesPresented',
      'presentCount',
      'droppedFrames',
      'drawCalls',
      'realPackDrawCount',
      'fallbackDrawCount',
      'skippedDrawCount',
      'skippedDraws',
      'unboundBindGroupCount',
      'validationErrors',
      'validationErrorCount',
      'computeDispatches',
      'packetsProcessed',
      'ringBytesDrained',
      'width',
      'height',
      'framesWithin30FpsBudget',
      'framesOver30FpsBudget',
    ]) {
      const val = pickNonNegInt(raw.frameStats, intKey);
      if (val !== undefined) fs[intKey] = val;
    }
    for (const numKey of [
      'lastFrameMs',
      'avgFrameMs',
      'fps',
      'presentFpsAvg',
      'onePercentLowFps',
      'fps1PctLow',
      'p95FrameMs',
      'p99FrameMs',
      'maxFrameMs',
      'targetFps',
      'budgetMs',
      'frameBudgetMs',
      'firstPresentMs',
      'firstVisibleFrameMs',
      'blackLoadingMs',
    ]) {
      const val = pickNonNegNumber(raw.frameStats, numKey);
      if (val !== undefined) fs[numKey] = val;
    }
    const fsBudgetCompliancePct = pickPct(raw.frameStats, 'budgetCompliancePct');
    if (fsBudgetCompliancePct !== undefined) fs.budgetCompliancePct = fsBudgetCompliancePct;
    const fsComplianceRate = pickUnitLevel(raw.frameStats, 'fps30ComplianceRate');
    if (fsComplianceRate !== undefined) fs.fps30ComplianceRate = fsComplianceRate;
    const fsDominantBottleneck = clipSanitizedString(raw.frameStats.dominantBottleneck, 64);
    if (fsDominantBottleneck !== undefined) fs.dominantBottleneck = fsDominantBottleneck;
    out.frameStats = fs;
  }

  const rawFallback = isPlainObject(raw.fallbackStats)
    ? raw.fallbackStats
    : isPlainObject(raw.shaderFallbackStats)
      ? raw.shaderFallbackStats
      : isPlainObject(raw.shaderFallback)
        ? raw.shaderFallback
        : null;
  if (rawFallback) {
    const fb = {};
    for (const intKey of [
      'packHits',
      'fallbackHits',
      'builtinHits',
      'totalLookups',
      'totalResolves',
      'uniqueMissing',
      'missingCount',
      'overflowMissingCount',
    ]) {
      const val = pickNonNegInt(rawFallback, intKey);
      if (val !== undefined) fb[intKey] = val;
    }
    const hitRatePct = pickPct(rawFallback, 'hitRatePct');
    if (hitRatePct !== undefined) fb.hitRatePct = hitRatePct;
    out.fallbackStats = fb;
  }

  const rawMissingList = Array.isArray(raw.missingShaders)
    ? raw.missingShaders
    : Array.isArray(raw.topMissingShaders)
      ? raw.topMissingShaders
      : isPlainObject(raw.missingShaderSummary) && Array.isArray(raw.missingShaderSummary.top)
        ? raw.missingShaderSummary.top
        : isPlainObject(raw.missingShaderSummary) && Array.isArray(raw.missingShaderSummary.entries)
          ? raw.missingShaderSummary.entries
          : null;

  if (rawMissingList) {
    const missingShaders = [];
    for (
      let i = 0;
      i < rawMissingList.length && missingShaders.length < MAX_GPU_MISSING_SHADERS;
      i += 1
    ) {
      const clean = sanitizeMissingShaderEntry(rawMissingList[i]);
      if (clean) {
        missingShaders.push(clean);
      }
    }
    out.missingShaders = missingShaders;
  }

  for (const binkKey of ['bink', 'binkPresenter', 'splashBridge']) {
    if (raw[binkKey] !== undefined && raw[binkKey] !== null) {
      const cleanBink = sanitizeBinkTelemetry(raw[binkKey]);
      if (cleanBink && Object.keys(cleanBink).length > 0) {
        out[binkKey] = cleanBink;
      }
    }
  }

  return out;
}

function sanitizeBootPhaseContext(raw) {
  if (raw === null || raw === undefined) {
    return null;
  }
  if (typeof raw === 'string') {
    const cleanPhase = clipSanitizedString(raw.trim(), 64);
    return cleanPhase ? { phase: cleanPhase } : null;
  }
  let source = raw;
  if (typeof source === 'object' && typeof source.getSummary === 'function') {
    try {
      source = source.getSummary();
    } catch {
      return null;
    }
  }
  if (!isPlainObject(source)) {
    return null;
  }

  const out = {};
  const phase =
    clipSanitizedString(
      typeof source.phase === 'string'
        ? source.phase.trim()
        : typeof source.bootPhase === 'string'
          ? source.bootPhase.trim()
          : typeof source.currentPhase === 'string'
            ? source.currentPhase.trim()
            : undefined,
      64
    );
  if (phase) out.phase = phase;
  const stage = clipSanitizedString(typeof source.stage === 'string' ? source.stage.trim() : undefined, 64);
  if (stage) out.stage = stage;
  const mode = clipSanitizedString(
    typeof source.mode === 'string'
      ? source.mode.trim()
      : typeof source.bootMode === 'string'
        ? source.bootMode.trim()
        : undefined,
    64
  );
  if (mode) out.mode = mode;
  const status = clipSanitizedString(typeof source.status === 'string' ? source.status.trim() : undefined, 64);
  if (status) out.status = status;

  const progressPct = pickPct(source, 'progressPct');
  if (progressPct !== undefined) out.progressPct = progressPct;
  const elapsedMs = pickNonNegNumber(source, 'elapsedMs');
  if (elapsedMs !== undefined) out.elapsedMs = elapsedMs;
  const phaseElapsedMs = pickNonNegNumber(source, 'phaseElapsedMs');
  if (phaseElapsedMs !== undefined) out.phaseElapsedMs = phaseElapsedMs;
  if (typeof source.readyForSplash === 'boolean') out.readyForSplash = source.readyForSplash;
  const milestonesReached = pickNonNegInt(source, 'milestonesReached');
  if (milestonesReached !== undefined) out.milestonesReached = milestonesReached;
  const trapCount = pickNonNegInt(source, 'trapCount');
  if (trapCount !== undefined) out.trapCount = trapCount;

  return Object.keys(out).length > 0 ? out : null;
}

function deriveBudgetCompliancePct(source) {
  if (!isPlainObject(source)) {
    return undefined;
  }
  const budgetObj = isPlainObject(source.budget) ? source.budget : null;
  const framesObj = isPlainObject(source.frames) ? source.frames : null;

  const explicitPct =
    pickPct(source, 'budgetCompliancePct') ??
    (budgetObj ? pickPct(budgetObj, 'budgetCompliancePct') : undefined) ??
    (framesObj ? pickPct(framesObj, 'budgetCompliancePct') : undefined);
  if (explicitPct !== undefined) {
    return roundTo2(explicitPct);
  }

  const within =
    pickNonNegInt(source, 'framesWithinBudget') ??
    pickNonNegInt(source, 'framesWithin30FpsBudget') ??
    (budgetObj ? pickNonNegInt(budgetObj, 'framesWithin30FpsBudget') : undefined) ??
    (framesObj ? pickNonNegInt(framesObj, 'framesWithin30FpsBudget') : undefined);
  const over =
    pickNonNegInt(source, 'framesOverBudget') ??
    pickNonNegInt(source, 'framesOver30FpsBudget') ??
    (budgetObj ? pickNonNegInt(budgetObj, 'framesOver30FpsBudget') : undefined) ??
    (framesObj ? pickNonNegInt(framesObj, 'framesOver30FpsBudget') : undefined);

  if (within !== undefined && over !== undefined) {
    const total = within + over;
    if (total > 0) {
      return roundTo2(Math.min(100, Math.max(0, (within / total) * 100)));
    }
    return undefined;
  }

  const rate =
    pickUnitLevel(source, 'fps30ComplianceRate') ??
    (budgetObj ? pickUnitLevel(budgetObj, 'fps30ComplianceRate') : undefined) ??
    (framesObj ? pickUnitLevel(framesObj, 'fps30ComplianceRate') : undefined);
  if (rate !== undefined) {
    return roundTo2(rate * 100);
  }
  return undefined;
}

/**
 * Strictly allowlists and sanitizes a 30 FPS frame-budget telemetry snapshot
 * (or duck-typed `PerfMonitor` instance via `getSnapshot()`).
 * Drops unknown keys, raw buffers, host paths, and secrets.
 *
 * @param {unknown} raw
 * @returns {object|null}
 */
function sanitizeFrameBudget(raw) {
  if (raw === null || raw === undefined) {
    return null;
  }
  let source = raw;
  if (typeof source === 'object' && typeof source.getSnapshot === 'function') {
    try {
      source = source.getSnapshot();
    } catch {
      return null;
    }
  }
  if (!isPlainObject(source)) {
    return null;
  }

  const budgetObj = isPlainObject(source.budget) ? source.budget : null;
  const framesObj = isPlainObject(source.frames) ? source.frames : null;
  const out = {};

  const explicitPresentFps =
    pickNonNegNumber(source, 'presentFpsAvg') ??
    (framesObj ? pickNonNegNumber(framesObj, 'presentFpsAvg') : undefined);
  if (explicitPresentFps !== undefined) {
    out.presentFpsAvg = roundTo2(explicitPresentFps);
  } else {
    const derivedFps =
      (framesObj && (pickNonNegInt(framesObj, 'totalFrames') ?? 1) > 0
        ? pickNonNegNumber(framesObj, 'fps')
        : undefined) ?? pickNonNegNumber(source, 'fps');
    if (derivedFps !== undefined && (derivedFps > 0 || !framesObj)) {
      out.presentFpsAvg = roundTo2(derivedFps);
    }
  }

  const budgetCompliancePct = deriveBudgetCompliancePct(source);
  if (budgetCompliancePct !== undefined) {
    out.budgetCompliancePct = budgetCompliancePct;
  }

  const targetFps =
    pickNonNegNumber(source, 'targetFps') ??
    (budgetObj ? pickNonNegNumber(budgetObj, 'targetFps') : undefined) ??
    (framesObj ? pickNonNegNumber(framesObj, 'targetFps') : undefined);
  if (targetFps !== undefined) out.targetFps = targetFps;

  const budgetMs =
    pickNonNegNumber(source, 'budgetMs') ??
    pickNonNegNumber(source, 'frameBudgetMs') ??
    (budgetObj ? pickNonNegNumber(budgetObj, 'budgetMs') : undefined) ??
    (framesObj ? pickNonNegNumber(framesObj, 'budgetMs') : undefined);
  if (budgetMs !== undefined) out.budgetMs = budgetMs;

  const framesWithinBudget =
    pickNonNegInt(source, 'framesWithinBudget') ??
    pickNonNegInt(source, 'framesWithin30FpsBudget') ??
    (budgetObj ? pickNonNegInt(budgetObj, 'framesWithin30FpsBudget') : undefined) ??
    (framesObj ? pickNonNegInt(framesObj, 'framesWithin30FpsBudget') : undefined);
  if (framesWithinBudget !== undefined) out.framesWithinBudget = framesWithinBudget;

  const framesOverBudget =
    pickNonNegInt(source, 'framesOverBudget') ??
    pickNonNegInt(source, 'framesOver30FpsBudget') ??
    (budgetObj ? pickNonNegInt(budgetObj, 'framesOver30FpsBudget') : undefined) ??
    (framesObj ? pickNonNegInt(framesObj, 'framesOver30FpsBudget') : undefined);
  if (framesOverBudget !== undefined) out.framesOverBudget = framesOverBudget;

  const onePercentLowFps =
    pickNonNegNumber(source, 'onePercentLowFps') ??
    pickNonNegNumber(source, 'fps1PctLow') ??
    (framesObj
      ? pickNonNegNumber(framesObj, 'onePercentLowFps') ?? pickNonNegNumber(framesObj, 'fps1PctLow')
      : undefined);
  if (onePercentLowFps !== undefined) out.onePercentLowFps = roundTo2(onePercentLowFps);

  const p95FrameMs =
    pickNonNegNumber(source, 'p95FrameMs') ??
    pickNonNegNumber(source, 'p95FrameTimeMs') ??
    (framesObj
      ? pickNonNegNumber(framesObj, 'p95FrameTimeMs') ?? pickNonNegNumber(framesObj, 'p95FrameMs')
      : undefined);
  if (p95FrameMs !== undefined) out.p95FrameMs = p95FrameMs;

  const p99FrameMs =
    pickNonNegNumber(source, 'p99FrameMs') ??
    pickNonNegNumber(source, 'p99FrameTimeMs') ??
    (framesObj
      ? pickNonNegNumber(framesObj, 'p99FrameTimeMs') ?? pickNonNegNumber(framesObj, 'p99FrameMs')
      : undefined);
  if (p99FrameMs !== undefined) out.p99FrameMs = p99FrameMs;

  const dominantBottleneck = clipSanitizedString(
    source.dominantBottleneck ??
      (budgetObj ? budgetObj.dominantBottleneck : undefined) ??
      (framesObj ? framesObj.dominantBottleneck : undefined),
    64
  );
  if (dominantBottleneck !== undefined) out.dominantBottleneck = dominantBottleneck;

  return out;
}

function resolveFrameBudgetContext({
  explicitFrameBudget,
  storedFrameBudget,
  explicitPresentFpsAvg,
  explicitBudgetCompliancePct,
  perfSnapshot,
  gpuTelemetry,
  bootSnapshot,
}) {
  const gpuFs = gpuTelemetry && isPlainObject(gpuTelemetry.frameStats) ? gpuTelemetry.frameStats : null;
  const perfBudget = isPlainObject(perfSnapshot) ? sanitizeFrameBudget(perfSnapshot) : null;
  const gpuBudget = isPlainObject(gpuTelemetry) ? sanitizeFrameBudget(gpuTelemetry) : null;
  const gpuFsBudget = gpuFs ? sanitizeFrameBudget(gpuFs) : null;

  let base = null;
  if (explicitFrameBudget === null) {
    base = null;
  } else if (explicitFrameBudget !== undefined) {
    base = sanitizeFrameBudget(explicitFrameBudget);
  } else if (storedFrameBudget !== null) {
    base = sanitizeValue(storedFrameBudget);
  }

  let presentFpsAvg = null;
  if (explicitPresentFpsAvg === null) {
    presentFpsAvg = null;
  } else if (
    typeof explicitPresentFpsAvg === 'number' &&
    Number.isFinite(explicitPresentFpsAvg) &&
    explicitPresentFpsAvg >= 0
  ) {
    presentFpsAvg = roundTo2(explicitPresentFpsAvg);
  } else {
    const candidate =
      (base ? pickNonNegNumber(base, 'presentFpsAvg') : undefined) ??
      (gpuTelemetry ? pickNonNegNumber(gpuTelemetry, 'presentFpsAvg') : undefined) ??
      (gpuFs ? pickNonNegNumber(gpuFs, 'presentFpsAvg') : undefined) ??
      (bootSnapshot ? pickNonNegNumber(bootSnapshot, 'presentFpsAvg') : undefined) ??
      (perfBudget ? pickNonNegNumber(perfBudget, 'presentFpsAvg') : undefined);
    presentFpsAvg = candidate !== undefined ? roundTo2(candidate) : null;
  }

  let budgetCompliancePct = null;
  if (explicitBudgetCompliancePct === null) {
    budgetCompliancePct = null;
  } else if (typeof explicitBudgetCompliancePct === 'number' && Number.isFinite(explicitBudgetCompliancePct)) {
    budgetCompliancePct = roundTo2(Math.min(100, Math.max(0, explicitBudgetCompliancePct)));
  } else {
    const candidate =
      (base ? pickPct(base, 'budgetCompliancePct') : undefined) ??
      (gpuBudget ? pickPct(gpuBudget, 'budgetCompliancePct') : undefined) ??
      (gpuFsBudget ? pickPct(gpuFsBudget, 'budgetCompliancePct') : undefined) ??
      (bootSnapshot ? pickPct(bootSnapshot, 'budgetCompliancePct') : undefined) ??
      (perfBudget ? pickPct(perfBudget, 'budgetCompliancePct') : undefined);
    budgetCompliancePct = candidate !== undefined ? roundTo2(candidate) : null;
  }

  let frameBudget = null;
  const hasExplicitOrStored = base !== null && Object.keys(base).length > 0;
  const hasAnyBudgetMetric =
    hasExplicitOrStored || presentFpsAvg !== null || budgetCompliancePct !== null;

  if (hasAnyBudgetMetric) {
    frameBudget = {};
    const mergeFrom = [perfBudget, gpuFsBudget, gpuBudget, bootSnapshot, base];
    for (let i = 0; i < mergeFrom.length; i += 1) {
      const src = mergeFrom[i];
      if (!isPlainObject(src)) continue;
      for (const key of [
        'targetFps',
        'budgetMs',
        'framesWithinBudget',
        'framesOverBudget',
        'onePercentLowFps',
        'p95FrameMs',
        'p99FrameMs',
        'dominantBottleneck',
      ]) {
        if (src[key] !== undefined) {
          frameBudget[key] = src[key];
        }
      }
    }
    if (presentFpsAvg !== null) {
      frameBudget.presentFpsAvg = presentFpsAvg;
    } else if (explicitPresentFpsAvg === null) {
      delete frameBudget.presentFpsAvg;
    }
    if (budgetCompliancePct !== null) {
      frameBudget.budgetCompliancePct = budgetCompliancePct;
    } else if (explicitBudgetCompliancePct === null) {
      delete frameBudget.budgetCompliancePct;
    }
    if (Object.keys(frameBudget).length === 0) {
      frameBudget = null;
    }
  }

  return {
    presentFpsAvg,
    budgetCompliancePct,
    frameBudget,
  };
}

function resolveBootPhaseState({
  explicitBootPhase,
  explicitBootPhaseContext,
  storedBootPhase,
  storedBootPhaseContext,
  bootSnapshot,
  perfSnapshot,
}) {
  let bootPhaseContext = null;
  if (explicitBootPhaseContext === null) {
    bootPhaseContext = null;
  } else if (explicitBootPhaseContext !== undefined) {
    bootPhaseContext = sanitizeBootPhaseContext(explicitBootPhaseContext);
  } else if (isPlainObject(explicitBootPhase)) {
    bootPhaseContext = sanitizeBootPhaseContext(explicitBootPhase);
  } else if (storedBootPhaseContext !== null) {
    bootPhaseContext = sanitizeValue(storedBootPhaseContext);
  }

  let bootPhase = null;
  if (explicitBootPhase === null) {
    bootPhase = null;
  } else if (typeof explicitBootPhase === 'string') {
    const cleaned = clipSanitizedString(explicitBootPhase.trim(), 64);
    bootPhase = cleaned || null;
  } else if (isPlainObject(explicitBootPhase) && bootPhaseContext) {
    bootPhase = bootPhaseContext.phase ?? bootPhaseContext.stage ?? null;
  } else if (storedBootPhase !== null) {
    bootPhase = storedBootPhase;
  } else if (bootPhaseContext && typeof bootPhaseContext.phase === 'string') {
    bootPhase = bootPhaseContext.phase;
  } else if (bootSnapshot && typeof bootSnapshot.bootPhase === 'string') {
    bootPhase = bootSnapshot.bootPhase;
  } else if (bootSnapshot && typeof bootSnapshot.phase === 'string') {
    bootPhase = bootSnapshot.phase;
  } else if (isPlainObject(perfSnapshot) && typeof perfSnapshot.currentPhase === 'string') {
    const cleaned = clipSanitizedString(perfSnapshot.currentPhase.trim(), 64);
    bootPhase = cleaned || null;
  }

  if (bootPhase !== null && bootPhaseContext === null && explicitBootPhaseContext !== null) {
    const ctx = { phase: bootPhase };
    if (bootSnapshot) {
      if (bootSnapshot.stage !== undefined) ctx.stage = bootSnapshot.stage;
      const mode = bootSnapshot.mode ?? bootSnapshot.bootMode;
      if (mode !== undefined) ctx.mode = mode;
      if (bootSnapshot.progressPct !== undefined) ctx.progressPct = bootSnapshot.progressPct;
      if (bootSnapshot.elapsedMs !== undefined) ctx.elapsedMs = bootSnapshot.elapsedMs;
      if (bootSnapshot.phaseElapsedMs !== undefined) ctx.phaseElapsedMs = bootSnapshot.phaseElapsedMs;
      if (bootSnapshot.readyForSplash !== undefined) ctx.readyForSplash = bootSnapshot.readyForSplash;
    }
    bootPhaseContext = ctx;
  } else if (bootPhase !== null && bootPhaseContext !== null && bootPhaseContext.phase === undefined) {
    bootPhaseContext = { phase: bootPhase, ...bootPhaseContext };
  }

  return {
    bootPhase,
    bootPhaseContext,
  };
}

function clipBoundedText(text, maxLen) {
  if (typeof text !== 'string') {
    return '';
  }
  return text.length > maxLen ? `${text.slice(0, maxLen - 3)}...` : text;
}

function extractErrorFields(errorOrMessage) {
  if (typeof errorOrMessage === 'string') {
    return {
      name: 'Error',
      message: clipBoundedText(sanitizeString(errorOrMessage), MAX_ERROR_MESSAGE_LENGTH),
      stack: null,
    };
  }

  if (errorOrMessage && typeof errorOrMessage === 'object') {
    const rawName =
      typeof errorOrMessage.name === 'string' && errorOrMessage.name.length > 0
        ? errorOrMessage.name
        : typeof errorOrMessage.reason === 'string' && !('stack' in errorOrMessage)
          ? 'GPUDeviceLostInfo'
          : errorOrMessage.constructor &&
              typeof errorOrMessage.constructor.name === 'string' &&
              errorOrMessage.constructor.name !== 'Object'
            ? errorOrMessage.constructor.name
            : 'Error';

    const rawMessage =
      typeof errorOrMessage.message === 'string'
        ? errorOrMessage.message
        : typeof errorOrMessage.reason === 'string'
          ? errorOrMessage.reason
          : String(errorOrMessage);

    const rawStack = typeof errorOrMessage.stack === 'string' ? errorOrMessage.stack : null;

    return {
      name: clipBoundedText(sanitizeString(rawName), 128),
      message: clipBoundedText(sanitizeString(rawMessage), MAX_ERROR_MESSAGE_LENGTH),
      stack: rawStack !== null ? clipBoundedText(sanitizeString(rawStack), MAX_ERROR_STACK_LENGTH) : null,
    };
  }

  return {
    name: 'Error',
    message: clipBoundedText(sanitizeString(errorOrMessage == null ? '' : String(errorOrMessage)), MAX_ERROR_MESSAGE_LENGTH),
    stack: null,
  };
}

function makeErrorCoalesceKey(category, message) {
  return `${category}\u0000${message}`;
}

class BoundedRing {
  constructor(capacity) {
    this._capacity = capacity;
    this._items = new Array(capacity);
    this._writeIdx = 0;
    this._count = 0;
    this._totalRecorded = 0;
  }

  get capacity() {
    return this._capacity;
  }

  get length() {
    return this._count;
  }

  get totalRecorded() {
    return this._totalRecorded;
  }

  get droppedCount() {
    return Math.max(0, this._totalRecorded - this._count);
  }

  push(item) {
    this.pushWithEvicted(item);
    return item;
  }

  pushWithEvicted(item) {
    const evicted = this._count === this._capacity ? this._items[this._writeIdx] : undefined;
    this._items[this._writeIdx] = item;
    this._writeIdx = (this._writeIdx + 1) % this._capacity;
    if (this._count < this._capacity) {
      this._count += 1;
    }
    this._totalRecorded += 1;
    return evicted;
  }

  toArray() {
    const out = new Array(this._count);
    const start = this._count < this._capacity ? 0 : this._writeIdx;
    for (let i = 0; i < this._count; i += 1) {
      out[i] = this._items[(start + i) % this._capacity];
    }
    return out;
  }

  clear() {
    this._items = new Array(this._capacity);
    this._writeIdx = 0;
    this._count = 0;
    this._totalRecorded = 0;
  }
}

/**
 * Captures Wasm traps, WebGPU validation/device-lost errors, worker/IO faults,
 * bounded recent runtime log lines, perf snapshots, and safe browser capability
 * fingerprints into a versioned, sanitized JSON diagnostic bundle.
 */
export class CrashReporter {
  /**
   * @param {object} [options]
   * @param {number} [options.maxLogs=200]
   * @param {number} [options.maxErrors=32]
   * @param {string} [options.appVersion='0.1.0']
   * @param {() => number} [options.now]
   * @param {boolean} [options.coalesceDuplicates=true]
   * @param {object} [options.navigatorObj]
   * @param {object} [options.globalObj]
   */
  constructor(options = {}) {
    if (options !== undefined && (options === null || typeof options !== 'object')) {
      throw new TypeError('CrashReporter options must be an object');
    }

    const {
      maxLogs = DEFAULT_MAX_LOGS,
      maxErrors = DEFAULT_MAX_ERRORS,
      appVersion = DEFAULT_APP_VERSION,
      now = () => Date.now(),
      coalesceDuplicates = true,
      navigatorObj = typeof navigator !== 'undefined' ? navigator : undefined,
      globalObj = typeof globalThis !== 'undefined' ? globalThis : undefined,
    } = options;

    if (!Number.isInteger(maxLogs) || maxLogs < 1) {
      throw new RangeError(`maxLogs must be a positive integer, got ${maxLogs}`);
    }
    if (!Number.isInteger(maxErrors) || maxErrors < 1) {
      throw new RangeError(`maxErrors must be a positive integer, got ${maxErrors}`);
    }
    if (typeof now !== 'function') {
      throw new TypeError('now must be a function returning epoch milliseconds');
    }
    if (typeof coalesceDuplicates !== 'boolean') {
      throw new TypeError('coalesceDuplicates must be a boolean');
    }

    this._maxLogs = maxLogs;
    this._maxErrors = maxErrors;
    this._appVersion = sanitizeString(appVersion || DEFAULT_APP_VERSION);
    this._now = now;
    this._coalesceDuplicates = coalesceDuplicates;
    this._navigatorObj = navigatorObj;
    this._globalObj = globalObj;

    this._logs = new BoundedRing(maxLogs);
    this._errors = new BoundedRing(maxErrors);
    this._errorIndex = new Map();
    this._recordedErrorCount = 0;
    this._coalescedErrorCount = 0;
    this._categoryCounts = Object.create(null);
    this._errorClassifications = new WeakMap();
    this._bootSnapshot = null;
    this._bootPhase = null;
    this._bootPhaseContext = null;
    this._frameBudget = null;
    this._gpuTelemetry = null;
    this._binkTelemetry = null;
    this._audioTelemetry = null;
    this._ioWorkerStats = null;
    this._runtimeDiagnostics = null;
    this._gpuAdapter = null;

    this._attachedTarget = null;
    this._boundOnError = null;
    this._boundOnRejection = null;
  }

  get maxLogs() {
    return this._maxLogs;
  }

  get maxErrors() {
    return this._maxErrors;
  }

  get appVersion() {
    return this._appVersion;
  }

  get logCount() {
    return this._logs.length;
  }

  get errorCount() {
    return this._errors.length;
  }

  get recordedErrorCount() {
    return this._recordedErrorCount;
  }

  get coalescedCount() {
    return this._coalescedErrorCount;
  }

  get categoryCounts() {
    return { ...this._categoryCounts };
  }

  get isGlobalHandlersAttached() {
    return this._attachedTarget !== null;
  }

  /**
   * Returns additive, sanitized error recording and coalescing statistics.
   *
   * @returns {{
   *   recordedCount: number,
   *   bufferedCount: number,
   *   coalescedCount: number,
   *   droppedCount: number,
   *   maxErrors: number,
   *   categoryCounts: Record<string, number>
   * }}
   */
  getErrorStats() {
    return {
      recordedCount: this._recordedErrorCount,
      bufferedCount: this._errors.length,
      coalescedCount: this._coalescedErrorCount,
      droppedCount: this._errors.droppedCount,
      maxErrors: this._maxErrors,
      categoryCounts: { ...this._categoryCounts },
    };
  }

  /**
   * Clears all buffered logs, errors, and attached boot/GPU snapshots without detaching global handlers.
   */
  clear() {
    this._logs.clear();
    this._errors.clear();
    this._errorIndex.clear();
    this._recordedErrorCount = 0;
    this._coalescedErrorCount = 0;
    this._categoryCounts = Object.create(null);
    this._bootSnapshot = null;
    this._bootPhase = null;
    this._bootPhaseContext = null;
    this._frameBudget = null;
    this._gpuTelemetry = null;
    this._binkTelemetry = null;
    this._audioTelemetry = null;
    this._ioWorkerStats = null;
    this._runtimeDiagnostics = null;
    this._gpuAdapter = null;
  }

  /**
   * Stores a sanitized boot phase string (e.g. `'boot'`, `'loading_assets'`, `'menu'`, `'gameplay'`)
   * or an allowlisted boot-phase context snapshot object to include in exported crash reports.
   * Pass `null` or `undefined` to clear.
   *
   * @param {string|object|null|undefined} phaseOrContext
   * @returns {string|object|null}
   */
  setBootPhase(phaseOrContext) {
    if (phaseOrContext === null || phaseOrContext === undefined) {
      this._bootPhase = null;
      this._bootPhaseContext = null;
      return null;
    }
    if (typeof phaseOrContext === 'string') {
      const cleanPhase = clipSanitizedString(phaseOrContext.trim(), 64);
      this._bootPhase = cleanPhase || null;
      this._bootPhaseContext = null;
      return this._bootPhase;
    }
    const cleanContext = sanitizeBootPhaseContext(phaseOrContext);
    this._bootPhaseContext = cleanContext;
    this._bootPhase = cleanContext ? cleanContext.phase ?? cleanContext.stage ?? null : null;
    return this.getBootPhaseContext();
  }

  /**
   * Returns the stored sanitized boot phase label, or `null` when unset.
   *
   * @returns {string|null}
   */
  getBootPhase() {
    return this._bootPhase;
  }

  /**
   * Returns a sanitized clone of the stored boot-phase context object, or `null` when unset.
   *
   * @returns {object|null}
   */
  getBootPhaseContext() {
    return this._bootPhaseContext ? sanitizeValue(this._bootPhaseContext) : null;
  }

  /**
   * Stores an allowlisted, sanitized 30 FPS frame-budget telemetry snapshot
   * (`presentFpsAvg`, `budgetCompliancePct`, target FPS / budget ms / bottleneck)
   * or duck-typed `PerfMonitor` instance via `getSnapshot()`.
   * Pass `null` or `undefined` to clear.
   *
   * @param {object|null|undefined} snapshot
   * @returns {object|null}
   */
  setFrameBudget(snapshot) {
    this._frameBudget = sanitizeFrameBudget(snapshot);
    return this.getFrameBudget();
  }

  /**
   * Returns a sanitized clone of the attached 30 FPS frame-budget snapshot, or `null` when unset.
   *
   * @returns {object|null}
   */
  getFrameBudget() {
    return this._frameBudget ? sanitizeValue(this._frameBudget) : null;
  }

  /**
   * Stores a sanitized lowercase WebGPU adapter label (max 96 chars, no host paths)
   * to include in exported diagnostic bundles. Pass `null` or `undefined` to clear.
   *
   * @param {string|object|null|undefined} adapter
   * @returns {string|null}
   */
  setGpuAdapter(adapter) {
    this._gpuAdapter = sanitizeGpuAdapter(adapter);
    return this._gpuAdapter;
  }

  /**
   * Returns the stored sanitized GPU adapter label, or `null` when unset.
   *
   * @returns {string|null}
   */
  getGpuAdapter() {
    return this._gpuAdapter;
  }

  /**
   * Stores an allowlisted, sanitized boot summary snapshot (or `BootTracer` instance)
   * to include under `bootDiagnostics.bootSnapshot` in exported diagnostic bundles.
   * Pass `null` or `undefined` to clear.
   *
   * @param {object|null|undefined} summary
   * @returns {object|null}
   */
  setBootSnapshot(summary) {
    this._bootSnapshot = sanitizeBootSnapshot(summary);
    return this.getBootSnapshot();
  }

  /**
   * Returns a sanitized clone of the attached boot snapshot, or `null` when unset.
   *
   * @returns {object|null}
   */
  getBootSnapshot() {
    return this._bootSnapshot ? sanitizeValue(this._bootSnapshot) : null;
  }

  /**
   * Stores an allowlisted, sanitized GPU telemetry snapshot to include under
   * `bootDiagnostics.gpuTelemetry` in exported diagnostic bundles.
   * Pass `null` or `undefined` to clear.
   *
   * @param {object|null|undefined} snapshot
   * @returns {object|null}
   */
  setGpuTelemetry(snapshot) {
    this._gpuTelemetry = sanitizeGpuTelemetry(snapshot);
    return this.getGpuTelemetry();
  }

  /**
   * Returns a sanitized clone of the attached GPU telemetry snapshot, or `null` when unset.
   *
   * @returns {object|null}
   */
  getGpuTelemetry() {
    return this._gpuTelemetry ? sanitizeValue(this._gpuTelemetry) : null;
  }

  /**
   * Stores an allowlisted, sanitized Bink2 / splash presenter telemetry snapshot
   * (or duck-typed `BinkVideoPresenter` / `SplashFrameBridge` instance via `getStats()`)
   * to include under `bootDiagnostics.binkTelemetry` in exported diagnostic bundles.
   * Pass `null` or `undefined` to clear.
   *
   * @param {object|null|undefined} snapshot
   * @returns {object|null}
   */
  setBinkTelemetry(snapshot) {
    this._binkTelemetry = sanitizeBinkTelemetry(snapshot);
    return this.getBinkTelemetry();
  }

  /**
   * Returns a sanitized clone of the attached Bink2 telemetry snapshot, or `null` when unset.
   *
   * @returns {object|null}
   */
  getBinkTelemetry() {
    return this._binkTelemetry ? sanitizeValue(this._binkTelemetry) : null;
  }

  /**
   * Stores an allowlisted, sanitized `#1027` guest audio PCM ring telemetry snapshot
   * (or duck-typed `AudioEngine` instance via `getStats()` / `getTelemetry()`)
   * to include under `bootDiagnostics.audioTelemetry` in exported diagnostic bundles.
   * Pass `null` or `undefined` to clear.
   *
   * @param {object|null|undefined} snapshot
   * @returns {object|null}
   */
  setAudioTelemetry(snapshot) {
    this._audioTelemetry = sanitizeAudioTelemetry(snapshot);
    return this.getAudioTelemetry();
  }

  /**
   * Returns a sanitized clone of the attached audio telemetry snapshot, or `null` when unset.
   *
   * @returns {object|null}
   */
  getAudioTelemetry() {
    return this._audioTelemetry ? sanitizeValue(this._audioTelemetry) : null;
  }

  /**
   * Stores an allowlisted, sanitized IO / HTTPFS worker telemetry snapshot
   * (`mailboxAttached`, HTTPFS read counts/bytes/in-flight/errors, cache hits) to include
   * under `bootDiagnostics.ioWorkerStats` in exported diagnostic bundles.
   * Pass `null` or `undefined` to clear.
   *
   * @param {object|null|undefined} snapshot
   * @returns {object|null}
   */
  setIoWorkerStats(snapshot) {
    this._ioWorkerStats = sanitizeIoWorkerStatsSnapshot(snapshot);
    return this.getIoWorkerStats();
  }

  /**
   * Returns a sanitized clone of the attached IO worker stats snapshot, or `null` when unset.
   *
   * @returns {object|null}
   */
  getIoWorkerStats() {
    return this._ioWorkerStats ? sanitizeValue(this._ioWorkerStats) : null;
  }

  /**
   * Stores an allowlisted, sanitized live runtime diagnostics snapshot (`dstorage`,
   * `blocked_waits`, `streaming_drain`) to include under `bootDiagnostics.runtimeDiagnostics`.
   * Pass `null` or `undefined` to clear.
   *
   * @param {object|null|undefined} snapshot
   * @returns {object|null}
   */
  setRuntimeDiagnostics(snapshot) {
    this._runtimeDiagnostics = sanitizeRuntimeDiagnosticsSnapshot(snapshot);
    return this.getRuntimeDiagnostics();
  }

  /**
   * Returns a sanitized clone of the attached runtime diagnostics snapshot, or `null` when unset.
   *
   * @returns {object|null}
   */
  getRuntimeDiagnostics() {
    return this._runtimeDiagnostics ? sanitizeValue(this._runtimeDiagnostics) : null;
  }

  /**
   * Appends a sanitized log entry into the bounded ring buffer.
   *
   * @param {string} level
   * @param {string} source
   * @param {unknown} message
   * @returns {{ timestamp: string, timestampMs: number, level: string, source: string, message: string }}
   */
  recordLog(level, source, message) {
    const ts = this._now();
    const entry = {
      timestamp: formatIsoTimestamp(ts),
      timestampMs: Number.isFinite(ts) ? ts : 0,
      level: normalizeLogLevel(level),
      source: clipBoundedText(sanitizeString(source || 'runtime'), 128),
      message: clipBoundedText(sanitizeString(message), MAX_LOG_MESSAGE_LENGTH),
    };
    this._logs.push(entry);
    return entry;
  }

  /**
   * Records and classifies a runtime error into the bounded error ring buffer.
   * When duplicate coalescing is enabled (default), repeated identical
   * `(category, message)` errors increment `occurrences` / `count` and update
   * `lastSeenAtMs` / `lastSeenMs` on the existing buffered entry instead of
   * evicting earlier root-cause errors or growing the JS heap unboundedly.
   *
   * @param {string|null|undefined} category
   * @param {unknown} errorOrMessage
   * @param {unknown} [context]
   * @returns {object}
   */
  recordError(category, errorOrMessage, context) {
    const ts = this._now();
    const safeTs = Number.isFinite(ts) ? ts : 0;
    const resolvedCategory = classifyError(errorOrMessage, category);
    const fields = extractErrorFields(errorOrMessage);

    this._recordedErrorCount += 1;
    this._categoryCounts[resolvedCategory] = (this._categoryCounts[resolvedCategory] || 0) + 1;

    const coalesceKey = makeErrorCoalesceKey(resolvedCategory, fields.message);
    if (this._coalesceDuplicates) {
      const existing = this._errorIndex.get(coalesceKey);
      if (existing) {
        existing.occurrences += 1;
        existing.count = existing.occurrences;
        const nextLastSeen = Math.max(existing.lastSeenAtMs, safeTs);
        existing.lastSeenAtMs = nextLastSeen;
        existing.lastSeenMs = nextLastSeen;
        this._coalescedErrorCount += 1;
        return { ...existing };
      }
    }

    let sanitizedContext = context !== undefined ? sanitizeValue(context) : null;
    if (
      errorOrMessage &&
      typeof errorOrMessage === 'object' &&
      typeof errorOrMessage.reason === 'string' &&
      !Array.isArray(sanitizedContext)
    ) {
      const reason = sanitizeString(errorOrMessage.reason);
      if (sanitizedContext && typeof sanitizedContext === 'object') {
        if (sanitizedContext.reason === undefined) {
          sanitizedContext = { ...sanitizedContext, reason };
        }
      } else {
        sanitizedContext = { reason };
      }
    }

    const entry = {
      timestamp: formatIsoTimestamp(ts),
      timestampMs: safeTs,
      firstSeenAtMs: safeTs,
      lastSeenAtMs: safeTs,
      firstSeenMs: safeTs,
      lastSeenMs: safeTs,
      occurrences: 1,
      count: 1,
      category: resolvedCategory,
      name: fields.name,
      message: fields.message,
      stack: fields.stack,
      context: sanitizedContext,
    };
    if (this._bootPhase !== null) {
      entry.bootPhase = this._bootPhase;
    }

    this._errorClassifications.set(entry, classifyBootError(errorOrMessage));
    const evicted = this._errors.pushWithEvicted(entry);
    if (evicted !== undefined) {
      const evictedKey = makeErrorCoalesceKey(evicted.category, evicted.message);
      if (this._errorIndex.get(evictedKey) === evicted) {
        this._errorIndex.delete(evictedKey);
      }
    }
    if (this._coalesceDuplicates) {
      this._errorIndex.set(coalesceKey, entry);
    }
    return { ...entry };
  }

  /**
   * Returns a shallow copy of buffered sanitized log entries (oldest to newest).
   */
  getLogs() {
    return this._logs.toArray();
  }

  /**
   * Returns a shallow copy of buffered sanitized error entries (oldest to newest).
   */
  getErrors() {
    return this._errors.toArray().map((entry) => ({ ...entry }));
  }

  /**
   * Safely fingerprints browser/runtime capabilities without leaking raw
   * user-agent strings, host paths, or cookies.
   *
   * @returns {object}
   */
  collectCapabilities() {
    const g = this._globalObj;
    const nav = this._navigatorObj;

    let crossOriginIsolated = false;
    try {
      crossOriginIsolated = Boolean(g && g.crossOriginIsolated === true);
    } catch {
      crossOriginIsolated = false;
    }

    let sharedArrayBuffer = false;
    try {
      sharedArrayBuffer = Boolean(g && typeof g.SharedArrayBuffer === 'function');
    } catch {
      sharedArrayBuffer = false;
    }

    let webgpu = false;
    try {
      webgpu = Boolean(nav && 'gpu' in nav && nav.gpu != null);
    } catch {
      webgpu = false;
    }

    let wasm64 = false;
    try {
      if (g && g.WebAssembly && typeof g.WebAssembly.Memory === 'function') {
        new g.WebAssembly.Memory({ initial: 1n, maximum: 2n, index: 'i64' });
        wasm64 = true;
      }
    } catch {
      wasm64 = false;
    }

    let opfs = false;
    try {
      opfs = Boolean(nav && nav.storage && typeof nav.storage.getDirectory === 'function');
    } catch {
      opfs = false;
    }

    let gamepad = false;
    try {
      gamepad = Boolean(nav && typeof nav.getGamepads === 'function');
    } catch {
      gamepad = false;
    }

    let audioWorklet = false;
    try {
      audioWorklet = Boolean(
        g &&
          (typeof g.AudioWorkletNode === 'function' ||
            typeof g.AudioWorklet === 'function' ||
            (typeof g.AudioContext === 'function' &&
              g.AudioContext.prototype &&
              'audioWorklet' in g.AudioContext.prototype))
      );
    } catch {
      audioWorklet = false;
    }

    let hardwareConcurrency = null;
    try {
      if (
        nav &&
        typeof nav.hardwareConcurrency === 'number' &&
        Number.isFinite(nav.hardwareConcurrency) &&
        nav.hardwareConcurrency > 0
      ) {
        hardwareConcurrency = Math.floor(nav.hardwareConcurrency);
      }
    } catch {
      hardwareConcurrency = null;
    }

    let deviceMemory = null;
    try {
      if (
        nav &&
        typeof nav.deviceMemory === 'number' &&
        Number.isFinite(nav.deviceMemory) &&
        nav.deviceMemory > 0
      ) {
        deviceMemory = nav.deviceMemory;
      }
    } catch {
      deviceMemory = null;
    }

    return {
      crossOriginIsolated,
      sharedArrayBuffer,
      webgpu,
      wasm64,
      opfs,
      gamepad,
      audioWorklet,
      hardwareConcurrency,
      deviceMemory,
    };
  }

  /**
   * Builds a versioned, recursively sanitized diagnostic report suitable for
   * JSON export or bug reports.
   *
   * @param {object} [options]
   * @param {object} [options.perfSnapshot] A `PerfMonitor.getSnapshot()` object or `PerfMonitor` instance.
   * @param {object} [options.storageSummary]
   * @param {object} [options.serverStatus]
   * @param {object} [options.bootSnapshot] Optional per-report boot summary override.
   * @param {object} [options.bootSummary] Alias for `bootSnapshot`.
   * @param {string|object|null} [options.bootPhase] Optional per-report boot-phase label or context override.
   * @param {object|null} [options.bootPhaseContext] Optional per-report boot-phase context snapshot override.
   * @param {object|null} [options.frameBudget] Optional per-report 30 FPS frame-budget snapshot override.
   * @param {number|null} [options.presentFpsAvg] Optional per-report average present FPS override.
   * @param {number|null} [options.budgetCompliancePct] Optional per-report 30 FPS budget compliance percentage override (0..100).
   * @param {object} [options.gpuTelemetry] Optional per-report GPU telemetry override.
   * @param {object} [options.binkTelemetry] Optional per-report Bink2/splash telemetry override.
   * @param {object} [options.audioTelemetry] Optional per-report `#1027` guest audio PCM ring telemetry override.
   * @param {string|object|null} [options.gpuAdapter] Optional per-report sanitized GPU adapter override.
   * @param {number|null} [options.firstVisibleFrameMs] Optional per-report first visible frame timestamp override (ms).
   * @param {number|null} [options.blackLoadingMs] Optional per-report black loading duration override (ms).
   * @param {unknown} [options.extra]
   * @returns {object}
   */
  buildReport({
    perfSnapshot,
    storageSummary,
    serverStatus,
    bootSnapshot,
    bootSummary,
    bootPhase,
    bootPhaseContext,
    frameBudget,
    presentFpsAvg,
    budgetCompliancePct,
    gpuTelemetry,
    binkTelemetry,
    audioTelemetry,
    ioWorkerStats,
    runtimeDiagnostics,
    gpuAdapter,
    firstVisibleFrameMs,
    blackLoadingMs,
    extra,
  } = {}) {
    const ts = this._now();

    let resolvedPerf = null;
    if (perfSnapshot && typeof perfSnapshot === 'object') {
      if (typeof perfSnapshot.getSnapshot === 'function') {
        resolvedPerf = perfSnapshot.getSnapshot();
      } else {
        resolvedPerf = perfSnapshot;
      }
    }

    const recentErrors = this._errors.toArray();
    const classifiedErrors = [];
    const errorCodeCounts = {};
    for (let i = 0; i < recentErrors.length; i += 1) {
      const errEntry = recentErrors[i];
      const classified = this._errorClassifications.get(errEntry) || classifyBootError(errEntry);
      const occurrences = errEntry.occurrences || errEntry.count || 1;
      classifiedErrors.push({
        timestamp: errEntry.timestamp,
        timestampMs: errEntry.timestampMs,
        firstSeenAtMs: errEntry.firstSeenAtMs,
        lastSeenAtMs: errEntry.lastSeenAtMs,
        firstSeenMs: errEntry.firstSeenMs ?? errEntry.firstSeenAtMs,
        lastSeenMs: errEntry.lastSeenMs ?? errEntry.lastSeenAtMs,
        occurrences,
        count: occurrences,
        errorCategory: errEntry.category,
        code: classified.code,
        category: classified.category,
        hint: classified.hint,
        message: classified.message,
      });
      errorCodeCounts[classified.code] = (errorCodeCounts[classified.code] || 0) + occurrences;
    }

    const rawBootOverride = bootSnapshot !== undefined ? bootSnapshot : bootSummary;
    const resolvedBootSnapshot =
      rawBootOverride !== undefined ? sanitizeBootSnapshot(rawBootOverride) : this._bootSnapshot;
    const resolvedGpuTelemetry =
      gpuTelemetry !== undefined ? sanitizeGpuTelemetry(gpuTelemetry) : this._gpuTelemetry;
    const resolvedBinkTelemetry =
      binkTelemetry !== undefined ? sanitizeBinkTelemetry(binkTelemetry) : this._binkTelemetry;
    const resolvedIoWorkerStats =
      ioWorkerStats !== undefined ? sanitizeIoWorkerStatsSnapshot(ioWorkerStats) : this._ioWorkerStats;
    const resolvedRuntimeDiagnostics =
      runtimeDiagnostics !== undefined
        ? sanitizeRuntimeDiagnosticsSnapshot(runtimeDiagnostics)
        : this._runtimeDiagnostics;

    let resolvedAudioTelemetry =
      audioTelemetry !== undefined ? sanitizeAudioTelemetry(audioTelemetry) : this._audioTelemetry;
    if (
      resolvedAudioTelemetry === null &&
      audioTelemetry === undefined &&
      isPlainObject(resolvedPerf) &&
      hasGuestAudioRingFields(resolvedPerf.audio)
    ) {
      resolvedAudioTelemetry = sanitizeAudioTelemetry(resolvedPerf.audio);
    }

    let resolvedGpuAdapter = null;
    if (gpuAdapter !== undefined) {
      resolvedGpuAdapter = sanitizeGpuAdapter(gpuAdapter);
    } else if (this._gpuAdapter !== null) {
      resolvedGpuAdapter = this._gpuAdapter;
    } else if (resolvedGpuTelemetry && typeof resolvedGpuTelemetry.gpuAdapter === 'string') {
      resolvedGpuAdapter = resolvedGpuTelemetry.gpuAdapter;
    } else if (resolvedBootSnapshot && typeof resolvedBootSnapshot.gpuAdapter === 'string') {
      resolvedGpuAdapter = resolvedBootSnapshot.gpuAdapter;
    } else if (resolvedGpuTelemetry && typeof resolvedGpuTelemetry.adapter === 'string') {
      resolvedGpuAdapter = sanitizeGpuAdapter(resolvedGpuTelemetry.adapter);
    }

    const { firstVisibleFrameMs: resolvedFirstVisibleFrameMs, blackLoadingMs: resolvedBlackLoadingMs } =
      resolveBootTimingMetrics({
        explicitFirstVisibleFrameMs: firstVisibleFrameMs,
        explicitBlackLoadingMs: blackLoadingMs,
        bootSnapshot: resolvedBootSnapshot,
        gpuTelemetry: resolvedGpuTelemetry,
      });

    const { bootPhase: resolvedBootPhase, bootPhaseContext: resolvedBootPhaseContext } =
      resolveBootPhaseState({
        explicitBootPhase: bootPhase,
        explicitBootPhaseContext: bootPhaseContext,
        storedBootPhase: this._bootPhase,
        storedBootPhaseContext: this._bootPhaseContext,
        bootSnapshot: resolvedBootSnapshot,
        perfSnapshot: resolvedPerf,
      });

    const {
      presentFpsAvg: resolvedPresentFpsAvg,
      budgetCompliancePct: resolvedBudgetCompliancePct,
      frameBudget: resolvedFrameBudget,
    } = resolveFrameBudgetContext({
      explicitFrameBudget: frameBudget,
      storedFrameBudget: this._frameBudget,
      explicitPresentFpsAvg: presentFpsAvg,
      explicitBudgetCompliancePct: budgetCompliancePct,
      perfSnapshot: resolvedPerf,
      gpuTelemetry: resolvedGpuTelemetry,
      bootSnapshot: resolvedBootSnapshot,
    });

    const report = {
      schemaVersion: SCHEMA_VERSION,
      generatedAt: formatIsoTimestamp(ts),
      appVersion: this._appVersion,
      capabilities: this.collectCapabilities(),
      errors: recentErrors,
      errorStats: this.getErrorStats(),
      recentLogs: this._logs.toArray(),
      perf: resolvedPerf,
      storage: storageSummary !== undefined ? storageSummary : null,
      serverStatus: serverStatus !== undefined ? serverStatus : null,
      bootDiagnostics: {
        classifiedErrors,
        errorCodeCounts,
        gpuAdapter: resolvedGpuAdapter,
        firstVisibleFrameMs: resolvedFirstVisibleFrameMs,
        blackLoadingMs: resolvedBlackLoadingMs,
        bootPhase: resolvedBootPhase,
        bootPhaseContext: resolvedBootPhaseContext,
        presentFpsAvg: resolvedPresentFpsAvg,
        budgetCompliancePct: resolvedBudgetCompliancePct,
        frameBudget: resolvedFrameBudget,
        bootSnapshot: resolvedBootSnapshot,
        gpuTelemetry: resolvedGpuTelemetry,
        binkTelemetry: resolvedBinkTelemetry,
        audioTelemetry: resolvedAudioTelemetry,
        ioWorkerStats: resolvedIoWorkerStats,
        runtimeDiagnostics: resolvedRuntimeDiagnostics,
      },
    };

    if (extra !== undefined) {
      report.extra = extra;
    }

    const sanitizedReport = sanitizeValue(report);
    if (
      sanitizedReport &&
      sanitizedReport.perf &&
      isPlainObject(sanitizedReport.perf.frames) &&
      sanitizedReport.perf.frames.onePercentLowFps === undefined
    ) {
      const explicitLow = pickNonNegNumber(sanitizedReport.perf.frames, 'fps1PctLow');
      if (explicitLow !== undefined) {
        sanitizedReport.perf.frames.onePercentLowFps = explicitLow;
      } else {
        const p99Ms = pickNonNegNumber(sanitizedReport.perf.frames, 'p99FrameTimeMs');
        if (p99Ms !== undefined && p99Ms > 0) {
          sanitizedReport.perf.frames.onePercentLowFps = roundTo2(1000 / p99Ms);
        }
      }
    }

    return sanitizedReport;
  }

  /**
   * Serializes `buildReport(options)` into a sanitized JSON string.
   *
   * @param {object} [options]
   * @param {number|string} [space]
   * @returns {string}
   */
  toJson(options = {}, space) {
    const opts = options && typeof options === 'object' ? options : {};
    const indent =
      space !== undefined
        ? space
        : opts.pretty === false
          ? 0
          : typeof opts.space === 'number' || typeof opts.space === 'string'
            ? opts.space
            : 2;

    return JSON.stringify(this.buildReport(opts), null, indent);
  }

  /**
   * Formats `buildReport(options)` as a strictly sanitized plain-text diagnostic report
   * (zero host paths, UNC paths, `file://` URLs, usernames, or auth tokens).
   *
   * @param {object} [options]
   * @returns {string}
   */
  toText(options = {}) {
    const opts = options && typeof options === 'object' ? options : {};
    return formatTextReport(this.buildReport(opts), opts);
  }

  /**
   * Formats `buildReport(options)` as a compact, strictly sanitized Markdown snippet
   * ready to paste directly onto GitHub issue comments.
   *
   * @param {object} [options]
   * @returns {string}
   */
  toGitHubIssueMarkdown(options = {}) {
    const opts = options && typeof options === 'object' ? options : {};
    return formatGitHubIssueMarkdown(this.buildReport(opts), opts);
  }

  /**
   * Alias for `toGitHubIssueMarkdown(options)` producing a compact, strictly sanitized
   * Markdown diagnostic summary.
   *
   * @param {object} [options]
   * @returns {string}
   */
  formatMarkdownReport(options = {}) {
    return this.toGitHubIssueMarkdown(options);
  }

  /**
   * Hooks global `error` and `unhandledrejection` events on an EventTarget (e.g. `window` or `self`).
   * Replaces any previously attached target to avoid duplicate handlers.
   *
   * @param {EventTarget} [target]
   * @returns {() => void} Detach function.
   */
  attachGlobalHandlers(target = this._globalObj) {
    if (
      !target ||
      typeof target.addEventListener !== 'function' ||
      typeof target.removeEventListener !== 'function'
    ) {
      throw new TypeError('attachGlobalHandlers requires an EventTarget with addEventListener/removeEventListener');
    }

    this.detachGlobalHandlers();

    const onError = (event) => {
      const errObj =
        event && event.error !== undefined && event.error !== null
          ? event.error
          : {
              name: 'Error',
              message: event && typeof event.message === 'string' ? event.message : 'Unhandled error event',
            };

      const context = {};
      if (event && typeof event.filename === 'string' && event.filename.length > 0) {
        context.filename = event.filename;
      }
      if (event && typeof event.lineno === 'number' && Number.isFinite(event.lineno)) {
        context.lineno = event.lineno;
      }
      if (event && typeof event.colno === 'number' && Number.isFinite(event.colno)) {
        context.colno = event.colno;
      }

      const hasContext = Object.keys(context).length > 0;
      const category = classifyError(errObj, null);
      this.recordError(category, errObj, hasContext ? context : undefined);
    };

    const onRejection = (event) => {
      const reason =
        event && event.reason !== undefined && event.reason !== null
          ? event.reason
          : 'Unhandled promise rejection';
      this.recordError('unhandled_rejection', reason);
    };

    target.addEventListener('error', onError);
    target.addEventListener('unhandledrejection', onRejection);

    this._attachedTarget = target;
    this._boundOnError = onError;
    this._boundOnRejection = onRejection;

    return () => {
      this.detachGlobalHandlers();
    };
  }

  /**
   * Removes previously attached global `error` and `unhandledrejection` listeners. Idempotent.
   */
  detachGlobalHandlers() {
    if (this._attachedTarget) {
      if (this._boundOnError) {
        this._attachedTarget.removeEventListener('error', this._boundOnError);
      }
      if (this._boundOnRejection) {
        this._attachedTarget.removeEventListener('unhandledrejection', this._boundOnRejection);
      }
    }
    this._attachedTarget = null;
    this._boundOnError = null;
    this._boundOnRejection = null;
  }
}

function formatCompactNumber(value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return null;
  }
  if (Number.isInteger(value)) {
    return String(value);
  }
  return String(roundTo2(value));
}

function escapeMdInlineCode(value, maxLen = 128) {
  const safe = sanitizeString(value == null ? '' : String(value))
    .replace(/[\r\n\t]+/g, ' ')
    .replace(/`/g, "'")
    .trim();
  const clipped = safe.length > maxLen ? `${safe.slice(0, maxLen - 3)}...` : safe;
  return `\`${clipped}\``;
}

function escapeMdTableCell(value, maxLen = 160) {
  const safe = sanitizeString(value == null ? '' : String(value))
    .replace(/[\r\n\t]+/g, ' ')
    .replace(/\|/g, '\\|')
    .trim();
  return safe.length > maxLen ? `${safe.slice(0, maxLen - 3)}...` : safe;
}

function resolveOnePercentLowFps(perf, gpu) {
  const frames = perf && isPlainObject(perf.frames) ? perf.frames : null;
  if (frames) {
    const explicit = pickNonNegNumber(frames, 'onePercentLowFps') ?? pickNonNegNumber(frames, 'fps1PctLow');
    if (explicit !== undefined) return explicit;
    const p99Ms = pickNonNegNumber(frames, 'p99FrameTimeMs');
    if (p99Ms !== undefined && p99Ms > 0) return roundTo2(1000 / p99Ms);
  }
  if (perf && isPlainObject(perf)) {
    const topExplicit = pickNonNegNumber(perf, 'onePercentLowFps') ?? pickNonNegNumber(perf, 'fps1PctLow');
    if (topExplicit !== undefined) return topExplicit;
  }
  if (gpu && isPlainObject(gpu)) {
    const gpuExplicit = pickNonNegNumber(gpu, 'onePercentLowFps') ?? pickNonNegNumber(gpu, 'fps1PctLow');
    if (gpuExplicit !== undefined) return gpuExplicit;
    const gpuP99 = pickNonNegNumber(gpu, 'p99FrameMs');
    if (gpuP99 !== undefined && gpuP99 > 0) return roundTo2(1000 / gpuP99);
    const fs = isPlainObject(gpu.frameStats) ? gpu.frameStats : null;
    if (fs) {
      const fsExplicit = pickNonNegNumber(fs, 'onePercentLowFps') ?? pickNonNegNumber(fs, 'fps1PctLow');
      if (fsExplicit !== undefined) return fsExplicit;
      const fsP99 = pickNonNegNumber(fs, 'p99FrameMs');
      if (fsP99 !== undefined && fsP99 > 0) return roundTo2(1000 / fsP99);
    }
  }
  return null;
}

function resolveTrapOffsetDisplay(trap) {
  if (!trap || !isPlainObject(trap)) {
    return null;
  }
  for (const key of ['instructionOffset', 'insnOffset', 'rva', 'funcOffset']) {
    const val = trap[key];
    if (val !== undefined && val !== null && val !== '') {
      return typeof val === 'number' ? `0x${val.toString(16)}` : String(val);
    }
  }
  return null;
}

function resolveTrapVaDisplay(trap) {
  if (!trap || !isPlainObject(trap)) {
    return null;
  }
  for (const key of ['guestVa', 'va', 'rip', 'pc', 'functionVa', 'funcVa']) {
    const val = trap[key];
    if (val !== undefined && val !== null && val !== '') {
      return String(val);
    }
  }
  return null;
}

/**
 * Formats a diagnostic report (or `CrashReporter` instance, or raw unsanitized
 * report-shaped object) into a compact, strictly sanitized Markdown snippet
 * ready to paste directly onto GitHub issues.
 *
 * Aggressively redacts local host paths, query secrets/tokens, and raw binary payloads
 * while preserving exact sanitized trap VAs (`0x...`), instruction offsets, GPU
 * validation/draw/skip counts, 1%-low FPS, Bink2 telemetry, and top missing shader hashes.
 *
 * @param {object|null|undefined} report
 * @param {object} [options]
 * @param {string} [options.title='### PlayGTA6Web Sanitized Diagnostic Summary']
 * @param {number} [options.maxTraps=5]
 * @param {number} [options.maxMissingShaders=5]
 * @param {number} [options.maxErrors=5]
 * @param {number} [options.maxUnresolvedImports=5]
 * @returns {string}
 */
export function formatGitHubIssueMarkdown(report, options = {}) {
  const opts = options && typeof options === 'object' ? options : {};
  const maxTraps = Number.isInteger(opts.maxTraps) && opts.maxTraps >= 0 ? opts.maxTraps : 5;
  const maxMissingShaders =
    Number.isInteger(opts.maxMissingShaders) && opts.maxMissingShaders >= 0 ? opts.maxMissingShaders : 5;
  const maxErrors = Number.isInteger(opts.maxErrors) && opts.maxErrors >= 0 ? opts.maxErrors : 5;
  const maxUnresolvedImports =
    Number.isInteger(opts.maxUnresolvedImports) && opts.maxUnresolvedImports >= 0
      ? opts.maxUnresolvedImports
      : 5;

  let rawSource = report;
  if (rawSource && typeof rawSource === 'object' && typeof rawSource.buildReport === 'function') {
    try {
      rawSource = rawSource.buildReport();
    } catch {
      rawSource = null;
    }
  }

  const safeReport = isPlainObject(rawSource) ? sanitizeValue(rawSource) : {};
  const rawBootDiag =
    isPlainObject(rawSource) && isPlainObject(rawSource.bootDiagnostics) ? rawSource.bootDiagnostics : null;

  const bootSnapshot = sanitizeBootSnapshot(
    rawBootDiag && rawBootDiag.bootSnapshot !== undefined
      ? rawBootDiag.bootSnapshot
      : isPlainObject(rawSource)
        ? rawSource.bootSnapshot ?? rawSource.bootSummary
        : null
  );
  const gpuTelemetry = sanitizeGpuTelemetry(
    rawBootDiag && rawBootDiag.gpuTelemetry !== undefined
      ? rawBootDiag.gpuTelemetry
      : isPlainObject(rawSource)
        ? rawSource.gpuTelemetry
        : null
  );
  const binkTelemetry = sanitizeBinkTelemetry(
    rawBootDiag && rawBootDiag.binkTelemetry !== undefined
      ? rawBootDiag.binkTelemetry
      : isPlainObject(rawSource) && rawSource.binkTelemetry !== undefined
        ? rawSource.binkTelemetry
        : gpuTelemetry
          ? gpuTelemetry.bink ?? gpuTelemetry.binkPresenter ?? gpuTelemetry.splashBridge ?? null
          : null
  );
  const rawPerfAudio =
    isPlainObject(rawSource) && isPlainObject(rawSource.perf) && hasGuestAudioRingFields(rawSource.perf.audio)
      ? rawSource.perf.audio
      : null;
  const audioTelemetry = sanitizeAudioTelemetry(
    rawBootDiag && rawBootDiag.audioTelemetry !== undefined
      ? rawBootDiag.audioTelemetry
      : isPlainObject(rawSource) && rawSource.audioTelemetry !== undefined
        ? rawSource.audioTelemetry
        : rawPerfAudio
  );

  const resolvedGpuAdapter =
    (rawBootDiag && rawBootDiag.gpuAdapter !== undefined ? sanitizeGpuAdapter(rawBootDiag.gpuAdapter) : null) ??
    (isPlainObject(rawSource) && rawSource.gpuAdapter !== undefined ? sanitizeGpuAdapter(rawSource.gpuAdapter) : null) ??
    (gpuTelemetry && typeof gpuTelemetry.gpuAdapter === 'string' ? gpuTelemetry.gpuAdapter : null) ??
    (bootSnapshot && typeof bootSnapshot.gpuAdapter === 'string' ? bootSnapshot.gpuAdapter : null) ??
    (gpuTelemetry && typeof gpuTelemetry.adapter === 'string' ? sanitizeGpuAdapter(gpuTelemetry.adapter) : null) ??
    null;

  const { firstVisibleFrameMs: resolvedFirstVisibleFrameMs, blackLoadingMs: resolvedBlackLoadingMs } =
    resolveBootTimingMetrics({
      explicitFirstVisibleFrameMs:
        rawBootDiag && rawBootDiag.firstVisibleFrameMs !== undefined
          ? rawBootDiag.firstVisibleFrameMs
          : isPlainObject(rawSource)
            ? rawSource.firstVisibleFrameMs
            : undefined,
      explicitBlackLoadingMs:
        rawBootDiag && rawBootDiag.blackLoadingMs !== undefined
          ? rawBootDiag.blackLoadingMs
          : isPlainObject(rawSource)
            ? rawSource.blackLoadingMs
            : undefined,
      bootSnapshot,
      gpuTelemetry,
    });

  const { bootPhase: resolvedBootPhase } = resolveBootPhaseState({
    explicitBootPhase:
      rawBootDiag && rawBootDiag.bootPhase !== undefined
        ? rawBootDiag.bootPhase
        : isPlainObject(rawSource)
          ? rawSource.bootPhase
          : undefined,
    explicitBootPhaseContext:
      rawBootDiag && rawBootDiag.bootPhaseContext !== undefined
        ? rawBootDiag.bootPhaseContext
        : isPlainObject(rawSource)
          ? rawSource.bootPhaseContext
          : undefined,
    storedBootPhase: null,
    storedBootPhaseContext: null,
    bootSnapshot,
    perfSnapshot: isPlainObject(safeReport.perf) ? safeReport.perf : null,
  });

  const {
    presentFpsAvg: resolvedPresentFpsAvg,
    budgetCompliancePct: resolvedBudgetCompliancePct,
    frameBudget: resolvedFrameBudget,
  } = resolveFrameBudgetContext({
    explicitFrameBudget:
      rawBootDiag && rawBootDiag.frameBudget !== undefined
        ? rawBootDiag.frameBudget
        : isPlainObject(rawSource)
          ? rawSource.frameBudget
          : undefined,
    storedFrameBudget: null,
    explicitPresentFpsAvg:
      rawBootDiag && rawBootDiag.presentFpsAvg !== undefined
        ? rawBootDiag.presentFpsAvg
        : isPlainObject(rawSource)
          ? rawSource.presentFpsAvg
          : undefined,
    explicitBudgetCompliancePct:
      rawBootDiag && rawBootDiag.budgetCompliancePct !== undefined
        ? rawBootDiag.budgetCompliancePct
        : isPlainObject(rawSource)
          ? rawSource.budgetCompliancePct
          : undefined,
    perfSnapshot: isPlainObject(safeReport.perf) ? safeReport.perf : null,
    gpuTelemetry,
    bootSnapshot,
  });

  const title =
    typeof opts.title === 'string' && opts.title.trim().length > 0
      ? sanitizeString(opts.title.trim())
      : '### PlayGTA6Web Sanitized Diagnostic Summary';

  const lines = [title];

  // Metadata line
  const metaParts = [];
  if (typeof safeReport.appVersion === 'string' && safeReport.appVersion.length > 0) {
    metaParts.push(`**App:** ${escapeMdInlineCode(safeReport.appVersion, 64)}`);
  }
  if (typeof safeReport.schemaVersion === 'number' && Number.isFinite(safeReport.schemaVersion)) {
    metaParts.push(`**Schema:** v${Math.floor(safeReport.schemaVersion)}`);
  }
  if (typeof safeReport.generatedAt === 'string' && safeReport.generatedAt.length > 0) {
    metaParts.push(`**Generated:** ${escapeMdInlineCode(safeReport.generatedAt, 64)}`);
  }
  if (metaParts.length > 0) {
    lines.push(`- ${metaParts.join(' | ')}`);
  }

  // Capabilities line
  if (isPlainObject(safeReport.capabilities)) {
    const c = safeReport.capabilities;
    const capParts = [];
    if (typeof c.wasm64 === 'boolean') capParts.push(`Wasm64=${c.wasm64 ? 'yes' : 'no'}`);
    if (typeof c.webgpu === 'boolean') capParts.push(`WebGPU=${c.webgpu ? 'yes' : 'no'}`);
    if (typeof c.sharedArrayBuffer === 'boolean') capParts.push(`SAB=${c.sharedArrayBuffer ? 'yes' : 'no'}`);
    if (typeof c.crossOriginIsolated === 'boolean') capParts.push(`COI=${c.crossOriginIsolated ? 'yes' : 'no'}`);
    if (typeof c.opfs === 'boolean') capParts.push(`OPFS=${c.opfs ? 'yes' : 'no'}`);
    if (typeof c.hardwareConcurrency === 'number' && c.hardwareConcurrency > 0) {
      capParts.push(`Threads=${Math.floor(c.hardwareConcurrency)}`);
    }
    if (capParts.length > 0) {
      lines.push(`- **Capabilities:** ${capParts.join(', ')}`);
    }
  }

  // Boot summary
  if (
    (bootSnapshot && Object.keys(bootSnapshot).length > 0) ||
    resolvedBootPhase !== null ||
    resolvedFirstVisibleFrameMs !== null ||
    resolvedBlackLoadingMs !== null
  ) {
    const bootParts = [];
    if (bootSnapshot && bootSnapshot.stage) bootParts.push(`stage=${escapeMdInlineCode(bootSnapshot.stage, 64)}`);
    if (resolvedBootPhase !== null && (!bootSnapshot || bootSnapshot.stage !== resolvedBootPhase)) {
      bootParts.push(`phase=${escapeMdInlineCode(resolvedBootPhase, 64)}`);
    }
    if (bootSnapshot && bootSnapshot.progressPct !== undefined) {
      bootParts.push(`progress=${formatCompactNumber(bootSnapshot.progressPct)}%`);
    }
    const mode = bootSnapshot ? bootSnapshot.mode || bootSnapshot.bootMode : undefined;
    if (mode) bootParts.push(`mode=${escapeMdInlineCode(mode, 64)}`);
    if (bootSnapshot && bootSnapshot.readyForSplash !== undefined) {
      bootParts.push(`readyForSplash=${bootSnapshot.readyForSplash ? 'yes' : 'no'}`);
    }
    if (resolvedFirstVisibleFrameMs !== null) {
      bootParts.push(`firstVisibleFrameMs=${formatCompactNumber(resolvedFirstVisibleFrameMs)}ms`);
    }
    if (resolvedBlackLoadingMs !== null) {
      bootParts.push(`blackLoadingMs=${formatCompactNumber(resolvedBlackLoadingMs)}ms`);
    }
    if (bootSnapshot && bootSnapshot.trapCount !== undefined) {
      const uniq =
        bootSnapshot.uniqueTrapCount !== undefined ? ` (unique=${bootSnapshot.uniqueTrapCount})` : '';
      bootParts.push(`traps=${bootSnapshot.trapCount}${uniq}`);
    }
    if (bootParts.length > 0) {
      lines.push(`- **Boot:** ${bootParts.join(', ')}`);
    }
  }

  // Performance / Frame pacing (including 1%-low FPS and 30 FPS budget)
  const perf = isPlainObject(safeReport.perf) ? safeReport.perf : null;
  const perfFrames = perf && isPlainObject(perf.frames) ? perf.frames : null;
  const gpuFrameStats = gpuTelemetry && isPlainObject(gpuTelemetry.frameStats) ? gpuTelemetry.frameStats : null;
  const fpsVal =
    (perfFrames && pickNonNegNumber(perfFrames, 'fps')) ??
    (gpuTelemetry && pickNonNegNumber(gpuTelemetry, 'fps')) ??
    (gpuFrameStats && pickNonNegNumber(gpuFrameStats, 'fps'));
  const onePctLowFps = resolveOnePercentLowFps(perf, gpuTelemetry);
  const avgMs =
    (perfFrames && pickNonNegNumber(perfFrames, 'avgFrameTimeMs')) ??
    (gpuFrameStats && pickNonNegNumber(gpuFrameStats, 'avgFrameMs'));
  const p95Ms =
    (perfFrames && pickNonNegNumber(perfFrames, 'p95FrameTimeMs')) ??
    (gpuTelemetry && pickNonNegNumber(gpuTelemetry, 'p95FrameMs')) ??
    (gpuFrameStats && pickNonNegNumber(gpuFrameStats, 'p95FrameMs'));
  const p99Ms =
    (perfFrames && pickNonNegNumber(perfFrames, 'p99FrameTimeMs')) ??
    (gpuTelemetry && pickNonNegNumber(gpuTelemetry, 'p99FrameMs')) ??
    (gpuFrameStats && pickNonNegNumber(gpuFrameStats, 'p99FrameMs'));
  const totalFrames = perfFrames ? pickNonNegInt(perfFrames, 'totalFrames') : undefined;
  const jankFrames = perfFrames ? pickNonNegInt(perfFrames, 'jankFrames') : undefined;

  const perfParts = [];
  if (fpsVal !== undefined) perfParts.push(`FPS=${escapeMdInlineCode(formatCompactNumber(fpsVal))}`);
  if (resolvedPresentFpsAvg !== null && (fpsVal === undefined || roundTo2(fpsVal) !== resolvedPresentFpsAvg)) {
    perfParts.push(`presentFpsAvg=${escapeMdInlineCode(formatCompactNumber(resolvedPresentFpsAvg))}`);
  }
  if (resolvedBudgetCompliancePct !== null) {
    perfParts.push(`budgetCompliancePct=${formatCompactNumber(resolvedBudgetCompliancePct)}%`);
  }
  if (resolvedFrameBudget && resolvedFrameBudget.dominantBottleneck) {
    perfParts.push(`bottleneck=${escapeMdInlineCode(resolvedFrameBudget.dominantBottleneck, 64)}`);
  }
  if (onePctLowFps !== null) {
    perfParts.push(`1%-low FPS=${escapeMdInlineCode(formatCompactNumber(onePctLowFps))}`);
  }
  if (avgMs !== undefined) perfParts.push(`avg=${formatCompactNumber(avgMs)}ms`);
  if (p95Ms !== undefined) perfParts.push(`p95=${formatCompactNumber(p95Ms)}ms`);
  if (p99Ms !== undefined) perfParts.push(`p99=${formatCompactNumber(p99Ms)}ms`);
  if (totalFrames !== undefined) perfParts.push(`frames=${totalFrames}`);
  if (jankFrames !== undefined) perfParts.push(`jank=${jankFrames}`);
  if (perfParts.length > 0) {
    lines.push(`- **Performance:** ${perfParts.join(', ')}`);
  }

  // GPU telemetry summary (adapter / validation / draw / skip counts + fallback stats)
  if ((gpuTelemetry && Object.keys(gpuTelemetry).length > 0) || resolvedGpuAdapter !== null) {
    const fs = gpuFrameStats || {};
    const gpuParts = [];
    if (gpuTelemetry && gpuTelemetry.backend) {
      gpuParts.push(`backend=${escapeMdInlineCode(gpuTelemetry.backend, 64)}`);
    }
    if (resolvedGpuAdapter !== null) {
      gpuParts.push(`gpuAdapter=${escapeMdInlineCode(resolvedGpuAdapter, MAX_GPU_ADAPTER_LENGTH)}`);
    }
    if (gpuTelemetry && gpuTelemetry.deviceLost === true) gpuParts.push('deviceLost=yes');

    const presents =
      (gpuTelemetry && (gpuTelemetry.framesPresented ?? gpuTelemetry.presentCount)) ??
      fs.framesPresented ??
      fs.presentCount;
    if (presents !== undefined) gpuParts.push(`presents=${presents}`);

    const drawCalls = (gpuTelemetry && gpuTelemetry.drawCalls) ?? fs.drawCalls;
    if (drawCalls !== undefined) gpuParts.push(`draws=${drawCalls}`);
    const realDraws = (gpuTelemetry && gpuTelemetry.realPackDrawCount) ?? fs.realPackDrawCount;
    if (realDraws !== undefined) gpuParts.push(`realPackDraws=${realDraws}`);
    const fallbackDraws = (gpuTelemetry && gpuTelemetry.fallbackDrawCount) ?? fs.fallbackDrawCount;
    if (fallbackDraws !== undefined) gpuParts.push(`fallbackDraws=${fallbackDraws}`);
    const skippedDraws =
      (gpuTelemetry && (gpuTelemetry.skippedDrawCount ?? gpuTelemetry.skippedDraws)) ??
      fs.skippedDrawCount ??
      fs.skippedDraws;
    if (skippedDraws !== undefined) gpuParts.push(`skippedDraws=${skippedDraws}`);
    const unboundBg = (gpuTelemetry && gpuTelemetry.unboundBindGroupCount) ?? fs.unboundBindGroupCount;
    if (unboundBg !== undefined) gpuParts.push(`unboundBindGroups=${unboundBg}`);

    const computeDispatches = (gpuTelemetry && gpuTelemetry.computeDispatches) ?? fs.computeDispatches;
    if (computeDispatches !== undefined) gpuParts.push(`compute=${computeDispatches}`);

    const validationErrors =
      (gpuTelemetry && (gpuTelemetry.validationErrorCount ?? gpuTelemetry.validationErrors)) ??
      fs.validationErrorCount ??
      fs.validationErrors;
    if (validationErrors !== undefined) gpuParts.push(`validationErrors=${validationErrors}`);

    if (gpuTelemetry && gpuTelemetry.fallbackStats) {
      const fb = gpuTelemetry.fallbackStats;
      if (fb.hitRatePct !== undefined) {
        gpuParts.push(`shaderHitRate=${formatCompactNumber(fb.hitRatePct)}%`);
      }
      if (fb.uniqueMissing !== undefined) {
        gpuParts.push(`uniqueMissingShaders=${fb.uniqueMissing}`);
      }
    }

    if (gpuParts.length > 0) {
      lines.push(`- **GPU:** ${gpuParts.join(', ')}`);
    }

    if (gpuTelemetry && (gpuTelemetry.lastValidationErrorCode || gpuTelemetry.lastValidationErrorMessage)) {
      const codeStr = gpuTelemetry.lastValidationErrorCode
        ? `${escapeMdInlineCode(gpuTelemetry.lastValidationErrorCode, 64)} `
        : '';
      const msgStr = gpuTelemetry.lastValidationErrorMessage
        ? escapeMdTableCell(gpuTelemetry.lastValidationErrorMessage, 200)
        : '';
      lines.push(`- **Last GPU Validation Error:** ${codeStr}${msgStr}`.trimEnd());
    }
  }

  // Bink2 / Splash video telemetry
  if (binkTelemetry && Object.keys(binkTelemetry).length > 0) {
    const presenter = isPlainObject(binkTelemetry.presenterStats) ? binkTelemetry.presenterStats : {};
    const binkParts = [];
    const fmt = binkTelemetry.pixelFormat ?? binkTelemetry.lastFrameFormat ?? presenter.pixelFormat;
    if (fmt) binkParts.push(`format=${escapeMdInlineCode(fmt, 32)}`);
    const width =
      binkTelemetry.frameWidth ?? binkTelemetry.lastFrameWidth ?? presenter.frameWidth ?? presenter.lastFrameWidth;
    const height =
      binkTelemetry.frameHeight ??
      binkTelemetry.lastFrameHeight ??
      presenter.frameHeight ??
      presenter.lastFrameHeight;
    if (width !== undefined && height !== undefined && width > 0 && height > 0) {
      binkParts.push(`res=${width}x${height}`);
    }
    const chroma = binkTelemetry.chromaOrder ?? binkTelemetry.lastChromaOrder ?? presenter.chromaOrder;
    if (chroma) binkParts.push(`chroma=${escapeMdInlineCode(chroma, 16)}`);
    const colorSpace = binkTelemetry.colorSpace ?? presenter.colorSpace;
    const colorRange = binkTelemetry.colorRange ?? presenter.colorRange;
    if (colorSpace) {
      binkParts.push(`color=${escapeMdInlineCode(colorRange ? `${colorSpace}/${colorRange}` : colorSpace, 32)}`);
    }
    const presented =
      binkTelemetry.presentedFrames ??
      binkTelemetry.framesPresented ??
      presenter.presentedFrames ??
      presenter.framesPresented;
    if (presented !== undefined) binkParts.push(`presented=${presented}`);
    const dropped = binkTelemetry.droppedFrames ?? presenter.droppedFrames;
    if (dropped !== undefined) binkParts.push(`dropped=${dropped}`);
    const skippedFrames = binkTelemetry.skippedFrames ?? presenter.skippedFrames;
    if (skippedFrames !== undefined) {
      binkParts.push(`skippedFrames=${skippedFrames}`);
    } else if (typeof binkTelemetry.skipped === 'boolean') {
      binkParts.push(`skipped=${binkTelemetry.skipped ? 'yes' : 'no'}`);
    }
    const uploaded = binkTelemetry.uploadedFrames ?? presenter.uploadedFrames;
    if (uploaded !== undefined) binkParts.push(`uploaded=${uploaded}`);
    const targetFps = binkTelemetry.targetFps ?? presenter.targetFps;
    if (targetFps !== undefined) binkParts.push(`targetFps=${formatCompactNumber(targetFps)}`);
    if (binkParts.length > 0) {
      lines.push(`- **Bink2:** ${binkParts.join(', ')}`);
    }
  }

  // #1027 Guest Audio PCM ring telemetry
  if (audioTelemetry && Object.keys(audioTelemetry).length > 0) {
    const audioParts = [];
    const outputState = audioTelemetry.outputState ?? audioTelemetry.contextState ?? audioTelemetry.state;
    if (outputState) {
      audioParts.push(`state=${escapeMdInlineCode(outputState, 32)}`);
    }
    if (typeof audioTelemetry.guestRingAttached === 'boolean') {
      audioParts.push(`guestRingAttached=${audioTelemetry.guestRingAttached ? 'yes' : 'no'}`);
    }
    if (audioTelemetry.guestFramesDrained !== undefined) {
      audioParts.push(`guestFramesDrained=${audioTelemetry.guestFramesDrained}`);
    }
    if (audioTelemetry.guestUnderruns !== undefined) {
      audioParts.push(`guestUnderruns=${audioTelemetry.guestUnderruns}`);
    }
    if (audioTelemetry.guestOverruns !== undefined) {
      audioParts.push(`guestOverruns=${audioTelemetry.guestOverruns}`);
    }
    if (audioTelemetry.guestPeakLevel !== undefined) {
      audioParts.push(`guestPeakLevel=${roundTo3(audioTelemetry.guestPeakLevel)}`);
    }
    if (audioParts.length > 0) {
      lines.push(`- **Audio:** ${audioParts.join(', ')}`);
    }
  }

  // IO / HTTPFS worker telemetry
  const ioWorkerStats = sanitizeIoWorkerStatsSnapshot(
    rawBootDiag && rawBootDiag.ioWorkerStats !== undefined
      ? rawBootDiag.ioWorkerStats
      : isPlainObject(rawSource)
        ? rawSource.ioWorkerStats
        : null
  );
  if (ioWorkerStats && Object.keys(ioWorkerStats).length > 0) {
    const ioParts = [];
    if (typeof ioWorkerStats.mailboxAttached === 'boolean') {
      ioParts.push(`mailboxAttached=${ioWorkerStats.mailboxAttached ? 'yes' : 'no'}`);
    }
    if (ioWorkerStats.mailboxLanes !== undefined) {
      ioParts.push(`lanes=${ioWorkerStats.mailboxLanes}`);
    }
    if (ioWorkerStats.readsCompleted !== undefined) {
      ioParts.push(`reads=${ioWorkerStats.readsCompleted}`);
    }
    if (ioWorkerStats.bytesTransferred !== undefined) {
      ioParts.push(`bytes=${ioWorkerStats.bytesTransferred}`);
    }
    if (ioWorkerStats.cacheHits !== undefined) {
      ioParts.push(`cacheHits=${ioWorkerStats.cacheHits}`);
    }
    if (ioWorkerStats.inFlightRequests !== undefined) {
      ioParts.push(`inFlight=${ioWorkerStats.inFlightRequests}`);
    }
    if (ioWorkerStats.errorCount !== undefined) {
      ioParts.push(`errors=${ioWorkerStats.errorCount}`);
    }
    if (ioParts.length > 0) {
      lines.push(`- **IO / HTTPFS:** ${ioParts.join(', ')}`);
    }
  }

  // Traps table (preserving exact guestVa and instruction offsets)
  if (bootSnapshot && maxTraps > 0) {
    const collectedTraps = [];
    const seenTrapKeys = new Set();
    const pushTrap = (trap) => {
      if (!isPlainObject(trap)) return;
      const va = resolveTrapVaDisplay(trap) || '';
      const off = resolveTrapOffsetDisplay(trap) || '';
      const key = `${trap.seq ?? ''}|${trap.kind ?? ''}|${va}|${off}|${trap.import ?? ''}|${trap.message ?? ''}`;
      if (seenTrapKeys.has(key)) return;
      seenTrapKeys.add(key);
      collectedTraps.push(trap);
    };
    if (bootSnapshot.lastTrap) {
      pushTrap(bootSnapshot.lastTrap);
    }
    if (Array.isArray(bootSnapshot.recentTraps)) {
      for (let i = bootSnapshot.recentTraps.length - 1; i >= 0 && collectedTraps.length < maxTraps; i -= 1) {
        pushTrap(bootSnapshot.recentTraps[i]);
      }
    }

    if (collectedTraps.length > 0) {
      lines.push('');
      lines.push('#### Traps');
      lines.push('| Kind | Guest VA | Insn Offset | Stage | Detail | Count |');
      lines.push('| --- | --- | --- | --- | --- | --- |');
      for (const trap of collectedTraps.slice(0, maxTraps)) {
        const kindCell = trap.kind ? escapeMdInlineCode(trap.kind, 48) : '-';
        const va = resolveTrapVaDisplay(trap);
        const vaCell = va ? escapeMdInlineCode(va, 32) : '-';
        const off = resolveTrapOffsetDisplay(trap);
        const offCell = off ? escapeMdInlineCode(off, 32) : '-';
        const stageCell = trap.stage ? escapeMdInlineCode(trap.stage, 48) : '-';
        const detailParts = [];
        if (trap.import) detailParts.push(escapeMdInlineCode(trap.import, 96));
        if (trap.mnemonic) detailParts.push(escapeMdInlineCode(trap.mnemonic, 32));
        if (trap.message) detailParts.push(escapeMdTableCell(trap.message, 140));
        const detailCell = detailParts.length > 0 ? detailParts.join(' ') : '-';
        const countCell = trap.count !== undefined ? String(trap.count) : '1';
        lines.push(`| ${kindCell} | ${vaCell} | ${offCell} | ${stageCell} | ${detailCell} | ${countCell} |`);
      }
    }

    // Top unresolved imports
    if (
      maxUnresolvedImports > 0 &&
      Array.isArray(bootSnapshot.unresolvedImports) &&
      bootSnapshot.unresolvedImports.length > 0
    ) {
      const sortedImports = [...bootSnapshot.unresolvedImports].sort(
        (a, b) => (b.count ?? 0) - (a.count ?? 0) || String(a.name ?? '').localeCompare(String(b.name ?? ''))
      );
      lines.push('');
      lines.push('#### Unresolved Imports');
      for (const imp of sortedImports.slice(0, maxUnresolvedImports)) {
        const label = imp.name || (imp.dll && imp.symbol ? `${imp.dll}!${imp.symbol}` : imp.symbol || 'unknown');
        const vaPart = imp.guestVa ? ` @ ${escapeMdInlineCode(imp.guestVa, 32)}` : '';
        const off = resolveTrapOffsetDisplay(imp);
        const offPart = off ? ` (+${escapeMdInlineCode(off, 32)})` : '';
        const countPart = imp.count !== undefined ? ` (x${imp.count})` : '';
        lines.push(`- ${escapeMdInlineCode(label, 128)}${vaPart}${offPart}${countPart}`);
      }
    }
  }

  // Top Missing Shaders table (preserving exact 0x... shader hashes)
  if (
    maxMissingShaders > 0 &&
    gpuTelemetry &&
    Array.isArray(gpuTelemetry.missingShaders) &&
    gpuTelemetry.missingShaders.length > 0
  ) {
    const sortedShaders = [...gpuTelemetry.missingShaders].sort(
      (a, b) => (b.hits ?? 0) - (a.hits ?? 0) || String(a.hash ?? '').localeCompare(String(b.hash ?? ''))
    );
    lines.push('');
    lines.push('#### Top Missing Shaders');
    lines.push('| Hash | Stage | Hits | First Frame |');
    lines.push('| --- | --- | --- | --- |');
    for (const shader of sortedShaders.slice(0, maxMissingShaders)) {
      const hashCell = shader.hash ? escapeMdInlineCode(shader.hash, 64) : '-';
      const stageCell = shader.stage ? escapeMdInlineCode(shader.stage, 32) : '-';
      const hitsCell = shader.hits !== undefined ? String(shader.hits) : '-';
      const frameCell = shader.firstSeenFrame !== undefined ? String(shader.firstSeenFrame) : '-';
      lines.push(`| ${hashCell} | ${stageCell} | ${hitsCell} | ${frameCell} |`);
    }
  }

  // Sanitized Errors summary
  const safeErrors = Array.isArray(safeReport.errors) ? safeReport.errors : [];
  const safeClassified =
    safeReport.bootDiagnostics && Array.isArray(safeReport.bootDiagnostics.classifiedErrors)
      ? safeReport.bootDiagnostics.classifiedErrors
      : [];

  if (maxErrors > 0 && safeErrors.length > 0) {
    lines.push('');
    lines.push('#### Errors');
    if (isPlainObject(safeReport.errorStats)) {
      const st = safeReport.errorStats;
      const statParts = [];
      if (typeof st.recordedCount === 'number') statParts.push(`recorded=${st.recordedCount}`);
      if (typeof st.coalescedCount === 'number') statParts.push(`coalesced=${st.coalescedCount}`);
      if (typeof st.droppedCount === 'number') statParts.push(`dropped=${st.droppedCount}`);
      if (statParts.length > 0) {
        lines.push(`- **Counts:** ${statParts.join(', ')}`);
      }
    }
    lines.push('| Category | Code | Occurrences | VA / Offset | Message |');
    lines.push('| --- | --- | --- | --- | --- |');
    for (let i = 0; i < safeErrors.length && i < maxErrors; i += 1) {
      const err = safeErrors[i];
      if (!isPlainObject(err)) continue;
      const classified = isPlainObject(safeClassified[i]) ? safeClassified[i] : classifyBootError(err);
      const catCell = err.category ? escapeMdInlineCode(err.category, 32) : '`runtime`';
      const codeCell = classified && classified.code ? escapeMdInlineCode(classified.code, 40) : '-';
      const occVal =
        typeof err.occurrences === 'number' && Number.isFinite(err.occurrences)
          ? err.occurrences
          : typeof err.count === 'number' && Number.isFinite(err.count)
            ? err.count
            : 1;
      const occCell = String(occVal);
      const ctx = isPlainObject(err.context) ? err.context : null;
      const ctxVa = resolveTrapVaDisplay(ctx);
      const ctxOff = resolveTrapOffsetDisplay(ctx);
      const locParts = [];
      if (ctxVa) locParts.push(escapeMdInlineCode(ctxVa, 32));
      if (ctxOff) locParts.push(escapeMdInlineCode(ctxOff, 32));
      const locCell = locParts.length > 0 ? locParts.join(' / ') : '-';
      const msgCell = escapeMdTableCell(err.message || err.name || '', 160) || '-';
      lines.push(`| ${catCell} | ${codeCell} | ${occCell} | ${locCell} | ${msgCell} |`);
    }
  }

  return sanitizeString(lines.join('\n'));
}

/**
 * Formats a diagnostic report (or `CrashReporter` instance, or raw unsanitized
 * report-shaped object) into a compact, strictly sanitized plain-text report
 * with zero host paths, UNC paths, `file://` URLs, usernames, or auth tokens.
 *
 * @param {object|null|undefined} report
 * @param {object} [options]
 * @param {string} [options.title='PlayGTA6Web Sanitized Crash Report']
 * @param {number} [options.maxErrors=10]
 * @param {number} [options.maxLogs=0]
 * @returns {string}
 */
export function formatTextReport(report, options = {}) {
  const opts = options && typeof options === 'object' ? options : {};
  const maxErrors = Number.isInteger(opts.maxErrors) && opts.maxErrors >= 0 ? opts.maxErrors : 10;
  const maxLogs = Number.isInteger(opts.maxLogs) && opts.maxLogs >= 0 ? opts.maxLogs : 0;

  let rawSource = report;
  if (rawSource && typeof rawSource === 'object' && typeof rawSource.buildReport === 'function') {
    try {
      rawSource = rawSource.buildReport();
    } catch {
      rawSource = null;
    }
  }

  const safeReport = isPlainObject(rawSource) ? sanitizeValue(rawSource) : {};
  const rawBootDiag =
    isPlainObject(rawSource) && isPlainObject(rawSource.bootDiagnostics) ? rawSource.bootDiagnostics : null;

  const bootSnapshot = sanitizeBootSnapshot(
    rawBootDiag && rawBootDiag.bootSnapshot !== undefined
      ? rawBootDiag.bootSnapshot
      : isPlainObject(rawSource)
        ? rawSource.bootSnapshot ?? rawSource.bootSummary
        : null
  );
  const gpuTelemetry = sanitizeGpuTelemetry(
    rawBootDiag && rawBootDiag.gpuTelemetry !== undefined
      ? rawBootDiag.gpuTelemetry
      : isPlainObject(rawSource)
        ? rawSource.gpuTelemetry
        : null
  );

  const { bootPhase: resolvedBootPhase } = resolveBootPhaseState({
    explicitBootPhase:
      rawBootDiag && rawBootDiag.bootPhase !== undefined
        ? rawBootDiag.bootPhase
        : isPlainObject(rawSource)
          ? rawSource.bootPhase
          : undefined,
    explicitBootPhaseContext:
      rawBootDiag && rawBootDiag.bootPhaseContext !== undefined
        ? rawBootDiag.bootPhaseContext
        : isPlainObject(rawSource)
          ? rawSource.bootPhaseContext
          : undefined,
    storedBootPhase: null,
    storedBootPhaseContext: null,
    bootSnapshot,
    perfSnapshot: isPlainObject(safeReport.perf) ? safeReport.perf : null,
  });

  const {
    presentFpsAvg: resolvedPresentFpsAvg,
    budgetCompliancePct: resolvedBudgetCompliancePct,
    frameBudget: resolvedFrameBudget,
  } = resolveFrameBudgetContext({
    explicitFrameBudget:
      rawBootDiag && rawBootDiag.frameBudget !== undefined
        ? rawBootDiag.frameBudget
        : isPlainObject(rawSource)
          ? rawSource.frameBudget
          : undefined,
    storedFrameBudget: null,
    explicitPresentFpsAvg:
      rawBootDiag && rawBootDiag.presentFpsAvg !== undefined
        ? rawBootDiag.presentFpsAvg
        : isPlainObject(rawSource)
          ? rawSource.presentFpsAvg
          : undefined,
    explicitBudgetCompliancePct:
      rawBootDiag && rawBootDiag.budgetCompliancePct !== undefined
        ? rawBootDiag.budgetCompliancePct
        : isPlainObject(rawSource)
          ? rawSource.budgetCompliancePct
          : undefined,
    perfSnapshot: isPlainObject(safeReport.perf) ? safeReport.perf : null,
    gpuTelemetry,
    bootSnapshot,
  });

  const title =
    typeof opts.title === 'string' && opts.title.trim().length > 0
      ? sanitizeString(opts.title.trim())
      : 'PlayGTA6Web Sanitized Crash Report';

  const lines = [title];
  const metaParts = [];
  if (typeof safeReport.appVersion === 'string' && safeReport.appVersion.length > 0) {
    metaParts.push(`app=${safeReport.appVersion}`);
  }
  if (typeof safeReport.schemaVersion === 'number' && Number.isFinite(safeReport.schemaVersion)) {
    metaParts.push(`schema=v${Math.floor(safeReport.schemaVersion)}`);
  }
  if (typeof safeReport.generatedAt === 'string' && safeReport.generatedAt.length > 0) {
    metaParts.push(`generatedAt=${safeReport.generatedAt}`);
  }
  if (metaParts.length > 0) {
    lines.push(`Meta: ${metaParts.join(' | ')}`);
  }

  const bootParts = [];
  if (bootSnapshot && bootSnapshot.stage) bootParts.push(`stage=${bootSnapshot.stage}`);
  if (resolvedBootPhase !== null) bootParts.push(`phase=${resolvedBootPhase}`);
  if (bootSnapshot && bootSnapshot.progressPct !== undefined) {
    bootParts.push(`progress=${formatCompactNumber(bootSnapshot.progressPct)}%`);
  }
  const mode = bootSnapshot ? bootSnapshot.mode || bootSnapshot.bootMode : undefined;
  if (mode) bootParts.push(`mode=${mode}`);
  if (bootParts.length > 0) {
    lines.push(`Boot: ${bootParts.join(', ')}`);
  }

  const budgetParts = [];
  if (resolvedPresentFpsAvg !== null) {
    budgetParts.push(`presentFpsAvg=${formatCompactNumber(resolvedPresentFpsAvg)}`);
  }
  if (resolvedBudgetCompliancePct !== null) {
    budgetParts.push(`budgetCompliancePct=${formatCompactNumber(resolvedBudgetCompliancePct)}%`);
  }
  if (resolvedFrameBudget && resolvedFrameBudget.targetFps !== undefined) {
    budgetParts.push(`targetFps=${formatCompactNumber(resolvedFrameBudget.targetFps)}`);
  }
  if (resolvedFrameBudget && resolvedFrameBudget.dominantBottleneck) {
    budgetParts.push(`bottleneck=${resolvedFrameBudget.dominantBottleneck}`);
  }
  if (budgetParts.length > 0) {
    lines.push(`FrameBudget: ${budgetParts.join(', ')}`);
  }

  const safeErrors = Array.isArray(safeReport.errors) ? safeReport.errors : [];
  if (maxErrors > 0 && safeErrors.length > 0) {
    lines.push(`Errors (${safeErrors.length}):`);
    for (let i = 0; i < safeErrors.length && i < maxErrors; i += 1) {
      const err = safeErrors[i];
      if (!isPlainObject(err)) continue;
      const count =
        typeof err.count === 'number' && Number.isFinite(err.count)
          ? err.count
          : typeof err.occurrences === 'number' && Number.isFinite(err.occurrences)
            ? err.occurrences
            : 1;
      const firstSeen =
        typeof err.firstSeenMs === 'number' && Number.isFinite(err.firstSeenMs)
          ? err.firstSeenMs
          : typeof err.firstSeenAtMs === 'number' && Number.isFinite(err.firstSeenAtMs)
            ? err.firstSeenAtMs
            : null;
      const lastSeen =
        typeof err.lastSeenMs === 'number' && Number.isFinite(err.lastSeenMs)
          ? err.lastSeenMs
          : typeof err.lastSeenAtMs === 'number' && Number.isFinite(err.lastSeenAtMs)
            ? err.lastSeenAtMs
            : null;
      const timing =
        firstSeen !== null && lastSeen !== null ? ` [firstSeenMs=${firstSeen} lastSeenMs=${lastSeen}]` : '';
      const category = err.category || 'runtime';
      const msg = String(err.message || err.name || '').replace(/[\r\n\t]+/g, ' ').trim();
      lines.push(`- [${category}] x${count}${timing} ${msg}`);
    }
  }

  const safeLogs = Array.isArray(safeReport.recentLogs) ? safeReport.recentLogs : [];
  if (maxLogs > 0 && safeLogs.length > 0) {
    lines.push(`RecentLogs (${Math.min(safeLogs.length, maxLogs)}):`);
    const slice = safeLogs.slice(-maxLogs);
    for (let i = 0; i < slice.length; i += 1) {
      const log = slice[i];
      if (!isPlainObject(log)) continue;
      const lvl = log.level || 'info';
      const src = log.source || 'runtime';
      const msg = String(log.message || '').replace(/[\r\n\t]+/g, ' ').trim();
      lines.push(`- [${lvl}] ${src}: ${msg}`);
    }
  }

  return sanitizeString(lines.join('\n'));
}

/**
 * Alias for `formatGitHubIssueMarkdown(report, options)` producing a compact,
 * strictly sanitized Markdown diagnostic summary.
 *
 * @param {object|null|undefined} report
 * @param {object} [options]
 * @returns {string}
 */
export function formatMarkdownReport(report, options = {}) {
  return formatGitHubIssueMarkdown(report, options);
}

export default CrashReporter;
