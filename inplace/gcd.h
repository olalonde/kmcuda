#pragma once

template <typename T>
void extended_gcd(T a, T b, T& gcd, T& mmi) {
  T x = 0;
  T lastx = 1;
  T y = 1;
  T lasty = 0;
  T origb = b;
  while (b != 0) {
    T quotient = a / b;
    T newb = a % b;
    a = b;
    b = newb;
    T newx = lastx - quotient * x;
    lastx = x;
    x = newx;
    T newy = lasty - quotient * y;
    lasty = y;
    y = newy;
  }
  gcd = a;
  mmi = 0;
  if (gcd == 1) {
    if (lastx < 0) {
        mmi = lastx + origb;
    } else {
        mmi = lastx;
    }
  }
}
