#include "test_harness.h"
#include "../src/loaders/hf_decode.h"
#include "../src/loaders/import_error.h"

void poly_test_torch_alloc_fail_after(int n);

TEST(torch_checkpoint, plan_contract) {
  FILE *f = fopen("test/fixtures/hf_plan.json", "rb");
  ASSERT_NOT_NULL(f);
  char data[8192];
  size_t len = fread(data, 1, sizeof(data), f);
  fclose(f);
  ASSERT_TRUE(len < sizeof(data));
  cJSON *cases = cJSON_ParseWithLength(data, len);
  ASSERT_NOT_NULL(cases);
  cJSON *row;
  cJSON_ArrayForEach(row, cases) {
    char *listing = cJSON_PrintUnformatted(cJSON_GetObjectItemCaseSensitive(row, "files"));
    cJSON *idx = cJSON_GetObjectItemCaseSensitive(row, "index");
    char *index =
        idx ? (cJSON_IsString(idx) ? strdup(idx->valuestring) : cJSON_PrintUnformatted(idx)) : NULL;
    cJSON *err = cJSON_GetObjectItemCaseSensitive(row, "error");
    int size = poly_hf_plan(listing, index, NULL, 0);
    if (err) {
      ASSERT_TRUE(size < 0);
      ASSERT_NOT_NULL(strstr(poly_import_last_error_message(), err->valuestring));
    } else {
      ASSERT_TRUE(size > 0);
      char *out = malloc((size_t)size);
      ASSERT_NOT_NULL(out);
      memset(out, '!', (size_t)size);
      ASSERT_EQ(poly_hf_plan(listing, index, out, size - 1), size);
      ASSERT_TRUE(out[0] == '!');
      ASSERT_EQ(poly_hf_plan(listing, index, out, size), size);
      cJSON *plan = cJSON_Parse(out);
      ASSERT_TRUE(cJSON_Compare(plan, cJSON_GetObjectItemCaseSensitive(row, "expected"), 1));
      cJSON_Delete(plan);
      free(out);
    }
    free(index);
    free(listing);
  }
  cJSON_Delete(cases);
  PASS();
}

static cJSON *torch_fixture(void) {
  FILE *f = fopen("test/fixtures/torch.json", "rb");
  if (!f) return NULL;
  char data[65536];
  size_t n = fread(data, 1, sizeof(data), f);
  bool ok = !ferror(f) && n < sizeof(data);
  fclose(f);
  return ok ? cJSON_ParseWithLength(data, n) : NULL;
}
static uint8_t *archive_bytes(cJSON *row, int64_t *len) {
  const char *s = cJSON_GetObjectItemCaseSensitive(row, "archive")->valuestring;
  const char *alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  uint8_t *bytes = malloc(strlen(s));
  if (!bytes) return NULL;
  unsigned bits = 0, acc = 0;
  *len = 0;
  for (; *s && *s != '='; s++) {
    const char *digit = strchr(alphabet, *s);
    if (!digit) {
      free(bytes);
      return NULL;
    }
    acc = acc << 6 | (unsigned)(digit - alphabet);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      bytes[(*len)++] = (uint8_t)(acc >> bits);
    }
  }
  return bytes;
}

TEST(torch_checkpoint, checkpoint_layouts_dtypes_and_aliases) {
  cJSON *fixture = torch_fixture();
  ASSERT_NOT_NULL(fixture);
  cJSON *row;
  cJSON_ArrayForEach(row, cJSON_GetObjectItemCaseSensitive(fixture, "cases")) {
    int64_t len;
    uint8_t *bytes = archive_bytes(row, &len);
    ASSERT_NOT_NULL(bytes);
    const uint8_t *files[] = {bytes};
    PolyHfDecoded *decoded = NULL;
    ASSERT_INT_EQ(poly_hf_decode("{}", 2, files, &len, 1, &decoded), 0);
    cJSON *expected = cJSON_GetObjectItemCaseSensitive(row, "tensors"), *item;
    ASSERT_INT_EQ(decoded->n_tensors, cJSON_GetArraySize(expected));
    int i = 0;
    cJSON_ArrayForEach(item, expected) {
      PolyDecodedTensor *t = &decoded->tensors[i++];
      ASSERT_TRUE(!strcmp(t->name, cJSON_GetObjectItemCaseSensitive(item, "name")->valuestring));
      ASSERT_INT_EQ(t->dtype, cJSON_GetObjectItemCaseSensitive(item, "dtype")->valueint);
      cJSON *shape = cJSON_GetObjectItemCaseSensitive(item, "shape");
      ASSERT_INT_EQ(t->ndim, cJSON_GetArraySize(shape));
      for (int j = 0; j < t->ndim; j++)
        ASSERT_INT_EQ(t->shape[j], cJSON_GetArrayItem(shape, j)->valueint);
      const char *hex = cJSON_GetObjectItemCaseSensitive(item, "hex")->valuestring;
      for (size_t j = 0; j < strlen(hex) / 2; j++) {
        unsigned byte;
        ASSERT_INT_EQ(sscanf(hex + j * 2, "%2x", &byte), 1);
        ASSERT_INT_EQ(((const uint8_t *)t->data)[j], byte);
      }
    }
    /* Distinct tensor names reference one contiguous storage, without a copy. */
    ASSERT_TRUE(decoded->tensors[0].data == decoded->tensors[6].data);
    poly_hf_decoded_free(decoded);
    free(bytes);
  }
  cJSON_Delete(fixture);
  PASS();
}

TEST(torch_checkpoint, allocation_failure_unwinds_without_publication) {
  cJSON *fixture = torch_fixture();
  ASSERT_NOT_NULL(fixture);
  int64_t len;
  uint8_t *bytes = archive_bytes(
      cJSON_GetArrayItem(cJSON_GetObjectItemCaseSensitive(fixture, "cases"), 0), &len
  );
  ASSERT_NOT_NULL(bytes);
  const uint8_t *files[] = {bytes};
  bool reached_success = false;
  for (int n = 0; n < 2000; n++) {
    poly_test_torch_alloc_fail_after(n);
    PolyHfDecoded *decoded = NULL;
    int rc = poly_hf_decode("{}", 2, files, &len, 1, &decoded);
    poly_test_torch_alloc_fail_after(-1);
    if (!rc) {
      poly_hf_decoded_free(decoded);
      reached_success = true;
      break;
    }
    ASSERT_TRUE(decoded == NULL);
    ASSERT_TRUE(strstr(poly_import_last_error_message(), "PyTorch checkpoint") != NULL);
  }
  ASSERT_TRUE(reached_success);
  free(bytes);
  cJSON_Delete(fixture);
  PASS();
}

TEST(torch_checkpoint, truncation_checksum_and_duplicate_shards) {
  cJSON *fixture = torch_fixture();
  ASSERT_NOT_NULL(fixture);
  int64_t len;
  uint8_t *bytes = archive_bytes(cJSON_GetObjectItemCaseSensitive(fixture, "model"), &len);
  ASSERT_NOT_NULL(bytes);
  const uint8_t *files[] = {bytes, bytes};
  PolyHfDecoded *decoded = NULL;
  for (int64_t truncated = 0; truncated < len; truncated += 13) {
    ASSERT_TRUE(poly_hf_decode("{}", 2, files, &truncated, 1, &decoded) != 0);
    ASSERT_TRUE(decoded == NULL);
  }
  int64_t lengths[] = {len, len};
  ASSERT_TRUE(poly_hf_decode("{}", 2, files, lengths, 2, &decoded) != 0);
  ASSERT_TRUE(decoded == NULL);
  /* The first local entry is data.pkl; corrupt it without updating the CRC. */
  bytes[64] ^= 1;
  ASSERT_TRUE(poly_hf_decode("{}", 2, files, &len, 1, &decoded) != 0);
  ASSERT_TRUE(decoded == NULL);
  free(bytes);
  cJSON_Delete(fixture);
  PASS();
}
