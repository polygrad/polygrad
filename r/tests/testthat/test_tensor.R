# Tests for the polygrad R Tensor class.

test_that("creation: from vector", {
  t <- Tensor$new(c(1, 2, 3))
  expect_equal(t$shape, 3L)
  expect_equal(t$as_numeric(), c(1, 2, 3), tolerance = 1e-5)
})

test_that("creation: from scalar", {
  t <- Tensor$new(42)
  expect_equal(t$as_numeric(), 42, tolerance = 1e-5)
})

test_that("creation: item", {
  t <- Tensor$new(42)
  expect_equal(t$item(), 42, tolerance = 1e-5)
})

test_that("elementwise: add", {
  a <- Tensor$new(c(1, 2, 3))
  b <- Tensor$new(c(4, 5, 6))
  c <- a + b
  expect_equal(c$as_numeric(), c(5, 7, 9), tolerance = 1e-5)
})

test_that("elementwise: sub", {
  a <- Tensor$new(c(10, 20, 30))
  b <- Tensor$new(c(1, 2, 3))
  expect_equal((a - b)$as_numeric(), c(9, 18, 27), tolerance = 1e-5)
})

test_that("elementwise: mul", {
  a <- Tensor$new(c(2, 3, 4))
  b <- Tensor$new(c(5, 6, 7))
  expect_equal((a * b)$as_numeric(), c(10, 18, 28), tolerance = 1e-5)
})

test_that("elementwise: div", {
  a <- Tensor$new(c(10, 20, 30))
  b <- Tensor$new(c(2, 4, 5))
  expect_equal((a / b)$as_numeric(), c(5, 5, 6), tolerance = 1e-5)
})

test_that("elementwise: neg", {
  a <- Tensor$new(c(1, -2, 3))
  expect_equal((-a)$as_numeric(), c(-1, 2, -3), tolerance = 1e-5)
})

test_that("elementwise: scalar add", {
  a <- Tensor$new(c(1, 2, 3))
  expect_equal((a + 2)$as_numeric(), c(3, 4, 5), tolerance = 1e-5)
})

test_that("elementwise: scalar mul", {
  a <- Tensor$new(c(1, 2, 3))
  expect_equal((a * 3)$as_numeric(), c(3, 6, 9), tolerance = 1e-5)
})

test_that("elementwise: exp2", {
  a <- Tensor$new(c(0, 1, 2, 3))
  expect_equal(a$exp2()$as_numeric(), c(1, 2, 4, 8), tolerance = 1e-5)
})

test_that("elementwise: sqrt", {
  a <- Tensor$new(c(1, 4, 9, 16))
  expect_equal(a$sqrt_op()$as_numeric(), c(1, 2, 3, 4), tolerance = 1e-5)
})

test_that("elementwise: chain", {
  a <- Tensor$new(c(1, 2, 3, 4))
  b <- Tensor$new(c(0.5, 0.5, 0.5, 0.5))
  c <- (a + Tensor$new(c(2, 2, 2, 2))) * b
  expect_equal(c$as_numeric(), c(1.5, 2, 2.5, 3), tolerance = 1e-5)
})

test_that("movement: flip", {
  a <- Tensor$new(c(1, 2, 3, 4, 5))
  expect_equal(a$flip_op(0)$as_numeric(), c(5, 4, 3, 2, 1), tolerance = 1e-5)
})

test_that("movement: pad", {
  a <- Tensor$new(c(1, 2, 3))
  b <- a$pad_op(list(c(1, 1)))
  expect_equal(b$shape, 5L)
  expect_equal(b$as_numeric(), c(0, 1, 2, 3, 0), tolerance = 1e-5)
})

test_that("reduce: sum all", {
  a <- Tensor$new(c(1, 2, 3, 4))
  expect_equal(a$sum_op()$item(), 10, tolerance = 1e-5)
})

test_that("autograd: grad mul sum", {
  x <- Tensor$new(c(1, 2, 3, 4), requires_grad_in = TRUE)
  loss <- (x * x)$sum_op()
  loss$backward()
  expect_false(is.null(x$grad))
  expect_equal(x$grad$as_numeric(), c(2, 4, 6, 8), tolerance = 1e-5)
})

test_that("autograd: grad neg sum", {
  x <- Tensor$new(c(1, 2, 3), requires_grad_in = TRUE)
  loss <- (-x)$sum_op()
  loss$backward()
  expect_equal(x$grad$as_numeric(), c(-1, -1, -1), tolerance = 1e-5)
})

test_that("stubs: matmul raises", {
  a <- Tensor$new(c(1, 2, 3))
  expect_error(a$matmul(a), "not implemented")
})

test_that("stubs: to raises", {
  a <- Tensor$new(1)
  expect_error(a$to("cuda"), "not implemented")
})

test_that("stubs: softmax raises", {
  a <- Tensor$new(c(1, 2, 3))
  expect_error(a$softmax(), "not implemented")
})

test_that("repr: show", {
  t <- Tensor$new(c(1, 2, 3))
  expect_output(t$show(), "shape=")
})
