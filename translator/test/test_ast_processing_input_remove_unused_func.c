
static void ToRemove() {
  return;
}

void NotToRemove1() {
  return;
}

static void NotToRemove2() {
  return;
}

void foo() {
  NotToRemove2();
}

static void PointerReferenced() {
  return;
}

void bar() {
  void (*x)() =  PointerReferenced;
  return;
}

