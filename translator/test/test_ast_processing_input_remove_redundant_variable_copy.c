
int DoesNotRemoveNonRedundantVariableCopy() {
  int x = 10;
  return x;
}

int RemovesRedundantVariableCopy() {
  int x = 10;
  int z __attribute__((unused)), y = x; // make sure only y is
                                        // removed and z is left as is
  return y;
}

int DoesNotRemoveVariableCopyWhenSrcReassigned() {
  int x = 10;
  int y = x;
  x = 20;
  return y;
}

int DoesNotRemoveVariableCopyWhenDstReassigned() {
  int x = 10;
  int y = x;
  y = 10;
  return y;
}

int foo(int x) {
  return x;
}

int RemoveRedundantVariableCopyWithFuncCall() {
  int x = 10;
  int y = x;
  return foo(y);
}

int RemoveWhenAssignedWithUnaryOp() {
  int x = 10;
  int y = -x;
  return foo(y);
}
