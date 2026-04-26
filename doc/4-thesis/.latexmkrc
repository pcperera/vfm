# Enable glossaries properly
add_cus_dep('acn', 'acr', 0, 'makeglossaries');

sub makeglossaries {
    system("makeglossaries \"$_[0]\"");
}

# Ensure enough passes
$max_repeat = 5;