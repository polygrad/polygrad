/*
 * test_wlrn.c -- Tests for poly_wlrn bundle reader
 */

#include "test_harness.h"
#include "../src/poly_wlrn.h"
#include <string.h>
#include <stdlib.h>

/* Helper: build a minimal WLRN bundle in memory */
static uint8_t *make_wlrn(const char *manifest, const char *toc,
                           const uint8_t *blob, uint32_t blob_len,
                           uint32_t *out_len) {
  uint32_t mlen = (uint32_t)strlen(manifest);
  uint32_t tlen = (uint32_t)strlen(toc);
  uint32_t total = 16 + mlen + tlen + blob_len;
  uint8_t *buf = calloc(total, 1);

  /* Magic "WLRN" LE */
  buf[0] = 'W'; buf[1] = 'L'; buf[2] = 'R'; buf[3] = 'N';
  /* Version 1 LE */
  buf[4] = 1; buf[5] = 0; buf[6] = 0; buf[7] = 0;
  /* Manifest length LE */
  buf[8] = mlen & 0xFF; buf[9] = (mlen >> 8) & 0xFF;
  buf[10] = (mlen >> 16) & 0xFF; buf[11] = (mlen >> 24) & 0xFF;
  /* TOC length LE */
  buf[12] = tlen & 0xFF; buf[13] = (tlen >> 8) & 0xFF;
  buf[14] = (tlen >> 16) & 0xFF; buf[15] = (tlen >> 24) & 0xFF;

  memcpy(buf + 16, manifest, mlen);
  memcpy(buf + 16 + mlen, toc, tlen);
  if (blob && blob_len > 0)
    memcpy(buf + 16 + mlen + tlen, blob, blob_len);

  *out_len = total;
  return buf;
}

TEST(wlrn, valid_bundle) {
  const char *manifest = "{\"typeId\":\"wlearn.nn.mlp@1\"}";
  const char *toc = "[{\"id\":\"weights\",\"offset\":0,\"length\":4}]";
  uint8_t blob[] = { 0xDE, 0xAD, 0xBE, 0xEF };
  uint32_t len;
  uint8_t *data = make_wlrn(manifest, toc, blob, 4, &len);

  PolyWlrnView v;
  int ret = poly_wlrn_view(data, len, &v);
  ASSERT_INT_EQ(ret, 0);

  ASSERT_INT_EQ(v.manifest_len, (int)strlen(manifest));
  ASSERT_TRUE(memcmp(v.manifest, manifest, v.manifest_len) == 0);

  ASSERT_INT_EQ(v.toc_len, (int)strlen(toc));
  ASSERT_TRUE(memcmp(v.toc, toc, v.toc_len) == 0);

  ASSERT_INT_EQ(v.blobs_len, 4);
  ASSERT_TRUE(memcmp(v.blobs, blob, 4) == 0);

  free(data);
  PASS();
}

TEST(wlrn, bad_magic) {
  uint8_t data[] = { 'X', 'Y', 'Z', 'W', 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  PolyWlrnView v;
  int ret = poly_wlrn_view(data, 16, &v);
  ASSERT_INT_EQ(ret, -1);
  PASS();
}

TEST(wlrn, too_short) {
  uint8_t data[] = { 'W', 'L', 'R', 'N' };
  PolyWlrnView v;
  int ret = poly_wlrn_view(data, 4, &v);
  ASSERT_INT_EQ(ret, -1);
  PASS();
}

TEST(wlrn, empty_blob) {
  const char *manifest = "{}";
  const char *toc = "[]";
  uint32_t len;
  uint8_t *data = make_wlrn(manifest, toc, NULL, 0, &len);

  PolyWlrnView v;
  int ret = poly_wlrn_view(data, len, &v);
  ASSERT_INT_EQ(ret, 0);
  ASSERT_INT_EQ(v.blobs_len, 0);

  free(data);
  PASS();
}

TEST(wlrn, truncated_header) {
  /* Manifest length claims 1000 bytes but total data is only 20 */
  uint8_t data[20] = { 'W', 'L', 'R', 'N', 1, 0, 0, 0,
                        0xE8, 0x03, 0, 0,   /* manifest_len = 1000 */
                        0, 0, 0, 0,
                        0, 0, 0, 0 };
  PolyWlrnView v;
  int ret = poly_wlrn_view(data, 20, &v);
  ASSERT_INT_EQ(ret, -1);
  PASS();
}
