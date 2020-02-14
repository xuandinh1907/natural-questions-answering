def get_add_tokens(do_enumerate):
    tags = ['Dd', 'Dl', 'Dt', 'H1', 'H2', 'H3', 'Li', 'Ol', 'P', 'Table', 'Td', 'Th', 'Tr', 'Ul']
    opening_tags = [f'<{tag}>' for tag in tags]
    closing_tags = [f'</{tag}>' for tag in tags]
    added_tags = opening_tags + closing_tags
    # See `nq_to_sqaud.py` for special-tokens
    special_tokens = ['<P>', '<Table>']
    if do_enumerate:
        for special_token in special_tokens:
            for j in range(11):
              added_tags.append(f'<{special_token[1: -1]}{j}>')

    add_tokens = ['Td_colspan', 'Th_colspan', '``', '\'\'', '--']
    add_tokens = add_tokens + added_tags
    return add_tokens