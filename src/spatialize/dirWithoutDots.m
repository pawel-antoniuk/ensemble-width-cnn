function items = dirWithoutDots(path)
items = dir(path);
items = items(~startsWith({items.name}, '.'));
