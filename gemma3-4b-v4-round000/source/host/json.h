// Minimal JSON reader for the engine protocol and safetensors headers (objects, arrays, strings, numbers,
// booleans, null). Numbers are kept as double (token ids and offsets are exact below 2^53).
#pragma once
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace json {

struct Value {
  enum Kind { Null, Bool, Num, Str, Arr, Obj } kind = Null;
  bool b = false;
  double num = 0;
  std::string str;
  std::vector<Value> arr;
  std::vector<std::pair<std::string, Value>> obj;
  bool has(const std::string& k) const {
    for (auto& kv : obj) if (kv.first == k) return true;
    return false;
  }
  const Value& at(const std::string& k) const {
    for (auto& kv : obj) if (kv.first == k) return kv.second;
    throw std::runtime_error("missing key " + k);
  }
};

struct Parser {
  const char* p;
  const char* e;
  void ws() { while (p < e && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++; }
  [[noreturn]] void fail(const char* m) { throw std::runtime_error(std::string("json: ") + m); }
  std::string string() {
    if (*p != '"') fail("expected string");
    p++;
    std::string s;
    while (p < e && *p != '"') {
      if (*p == '\\') {
        p++;
        if (p >= e) fail("bad escape");
        char c = *p++;
        switch (c) {
          case 'n': s += '\n'; break;
          case 't': s += '\t'; break;
          case 'r': s += '\r'; break;
          case 'b': s += '\b'; break;
          case 'f': s += '\f'; break;
          case 'u': {
            if (e - p < 4) fail("bad \\u");
            unsigned cp = (unsigned)strtoul(std::string(p, 4).c_str(), nullptr, 16);
            p += 4;
            if (cp < 0x80) s += (char)cp;
            else if (cp < 0x800) { s += (char)(0xC0 | (cp >> 6)); s += (char)(0x80 | (cp & 0x3F)); }
            else { s += (char)(0xE0 | (cp >> 12)); s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F)); }
            break;
          }
          default: s += c;
        }
      } else {
        s += *p++;
      }
    }
    if (p >= e) fail("unterminated string");
    p++;
    return s;
  }
  Value value() {
    ws();
    if (p >= e) fail("unexpected end");
    Value v;
    char c = *p;
    if (c == '{') {
      v.kind = Value::Obj;
      p++; ws();
      if (*p == '}') { p++; return v; }
      for (;;) {
        ws();
        std::string k = string();
        ws();
        if (*p != ':') fail("expected :");
        p++;
        v.obj.emplace_back(k, value());
        ws();
        if (*p == ',') { p++; continue; }
        if (*p == '}') { p++; break; }
        fail("expected , or }");
      }
    } else if (c == '[') {
      v.kind = Value::Arr;
      p++; ws();
      if (*p == ']') { p++; return v; }
      for (;;) {
        v.arr.push_back(value());
        ws();
        if (*p == ',') { p++; continue; }
        if (*p == ']') { p++; break; }
        fail("expected , or ]");
      }
    } else if (c == '"') {
      v.kind = Value::Str;
      v.str = string();
    } else if (c == 't' && e - p >= 4 && !strncmp(p, "true", 4)) {
      v.kind = Value::Bool; v.b = true; p += 4;
    } else if (c == 'f' && e - p >= 5 && !strncmp(p, "false", 5)) {
      v.kind = Value::Bool; v.b = false; p += 5;
    } else if (c == 'n' && e - p >= 4 && !strncmp(p, "null", 4)) {
      p += 4;
    } else {
      char* end;
      v.kind = Value::Num;
      v.num = strtod(p, &end);
      if (end == p) fail("bad value");
      p = end;
    }
    return v;
  }
};

inline Value parse(const std::string& s) {
  Parser ps{s.data(), s.data() + s.size()};
  Value v = ps.value();
  return v;
}

inline std::string escape(const std::string& s) {
  std::string o;
  for (char c : s) {
    if (c == '"' || c == '\\') { o += '\\'; o += c; }
    else if (c == '\n') o += "\\n";
    else if ((unsigned char)c < 0x20) o += ' ';
    else o += c;
  }
  return o;
}

}  // namespace json
