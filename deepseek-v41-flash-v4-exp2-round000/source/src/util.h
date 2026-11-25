// Small utilities: errors, timing, a minimal JSON reader (enough for safetensors headers and protocol requests).
#pragma once
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#define LOG(...) do { fprintf(stderr, "[engine] " __VA_ARGS__); fputc('\n', stderr); fflush(stderr); } while (0)

inline double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

[[noreturn]] inline void die(const std::string& m) { LOG("fatal: %s", m.c_str()); exit(1); }

// ------------------------------------------------------------------------------------------------- JSON
struct JVal {
  enum T { NUL, BOOL, NUM, STR, ARR, OBJ } t = NUL;
  bool b = false;
  double num = 0;
  int64_t inum = 0;
  bool is_int = false;
  std::string s;
  std::vector<JVal> a;
  std::vector<std::pair<std::string, JVal>> o;
  const JVal* get(const char* k) const {
    for (auto& kv : o) if (kv.first == k) return &kv.second;
    return nullptr;
  }
};

struct JParser {
  const char* p;
  const char* e;
  JParser(const char* b, size_t n) : p(b), e(b + n) {}
  void ws() { while (p < e && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) p++; }
  [[noreturn]] void err(const char* m) { throw std::runtime_error(std::string("json: ") + m); }
  JVal parse() {
    ws();
    if (p >= e) err("eof");
    JVal v;
    char c = *p;
    if (c == '{') {
      v.t = JVal::OBJ; p++; ws();
      if (p < e && *p == '}') { p++; return v; }
      while (true) {
        ws(); if (p >= e || *p != '"') err("key");
        std::string k = str();
        ws(); if (p >= e || *p != ':') err(":"); p++;
        v.o.emplace_back(std::move(k), parse());
        ws(); if (p < e && *p == ',') { p++; continue; }
        if (p < e && *p == '}') { p++; break; }
        err("obj");
      }
    } else if (c == '[') {
      v.t = JVal::ARR; p++; ws();
      if (p < e && *p == ']') { p++; return v; }
      while (true) {
        v.a.push_back(parse());
        ws(); if (p < e && *p == ',') { p++; continue; }
        if (p < e && *p == ']') { p++; break; }
        err("arr");
      }
    } else if (c == '"') {
      v.t = JVal::STR; v.s = str();
    } else if (c == 't' && e - p >= 4 && !strncmp(p, "true", 4)) { v.t = JVal::BOOL; v.b = true; p += 4; }
    else if (c == 'f' && e - p >= 5 && !strncmp(p, "false", 5)) { v.t = JVal::BOOL; v.b = false; p += 5; }
    else if (c == 'n' && e - p >= 4 && !strncmp(p, "null", 4)) { v.t = JVal::NUL; p += 4; }
    else {
      v.t = JVal::NUM;
      const char* s0 = p;
      bool fl = false;
      if (p < e && (*p == '-' || *p == '+')) p++;
      while (p < e && ((*p >= '0' && *p <= '9') || *p == '.' || *p == 'e' || *p == 'E' || *p == '-' || *p == '+')) {
        if (*p == '.' || *p == 'e' || *p == 'E') fl = true;
        p++;
      }
      if (p == s0) err("value");
      std::string t(s0, p);
      if (!fl) { v.is_int = true; v.inum = strtoll(t.c_str(), nullptr, 10); v.num = (double)v.inum; }
      else { v.num = strtod(t.c_str(), nullptr); v.inum = (int64_t)v.num; }
    }
    return v;
  }
  std::string str() {
    p++;  // "
    std::string r;
    while (p < e && *p != '"') {
      if (*p == '\\') {
        p++;
        if (p >= e) err("esc");
        char c = *p++;
        switch (c) {
          case 'n': r += '\n'; break; case 't': r += '\t'; break; case 'r': r += '\r'; break;
          case 'b': r += '\b'; break; case 'f': r += '\f'; break;
          case 'u': { if (e - p < 4) err("u"); r += '?'; p += 4; break; }
          default: r += c;
        }
      } else r += *p++;
    }
    if (p >= e) err("str");
    p++;
    return r;
  }
};

inline JVal json_parse(const std::string& s) { JParser jp(s.data(), s.size()); return jp.parse(); }

inline std::string json_escape(const std::string& s) {
  std::string r;
  for (char c : s) {
    if (c == '"' || c == '\\') { r += '\\'; r += c; }
    else if (c == '\n') r += "\\n";
    else if ((unsigned char)c < 0x20) r += ' ';
    else r += c;
  }
  return r;
}

// bf16 helpers
inline uint16_t f2bf_exact(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)(u >> 16); }
inline float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
