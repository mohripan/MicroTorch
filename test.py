from microTorch import MicroNode

node1 = MicroNode(10)
node2 = MicroNode(5)

sum_node = node1.add(node2)
diff_node = node1.sub(node2)
prod_node = node1.mul(node2)
quot_node = node1.div(node2)

sum_node = node1 + node2

print("Sum:", sum_node.data)
print("Difference:", diff_node.data)
print("Product:", prod_node.data)
print("Quotient:", quot_node.data)