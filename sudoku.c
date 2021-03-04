/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   sudoku.c                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: hchd <hchd@student.42seoul.kr>             +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2021/03/04 15:40:26 by hchd              #+#    #+#             */
/*   Updated: 2021/03/04 22:47:20 by hchd             ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <stdio.h>
#include <unistd.h>
#include <stdbool.h>

void	Erase_space(char *str)
{
	char	*dest;

	dest = str;
	while (*str)
	{
		if (*str != ' ')
		{
			if (dest != str)
			{
				*dest = *str;
			}
			dest++;
		}
		str++;
	}
	*dest = 0;
}

void	initialization(char *table)
{
	int i;
	int j;

	i = 0;
	j = 0;
	while (i < 4)
	{
		while (j < 4)
		{
			table[i][j] = '0';
			j++;
		}
		i++;
	}
}

void	record_1234(int index)
{
	int i;
	int j;

	i = 1;
	j = 1;
	if (index >= 8)
	{
		if (index < 12)
		{
			index = index % 4;
			while (j <= 4)
			{
				table[index][j] = j;
				j++;
			}
		}
		else
		{
			index = index % 4;
			while (j <= 4)
			{
				table[index][j] = 5 - j;
				j++;
			}
		}
	}
	else
	{
		if (index < 4)
		{
			index = index % 4;
			while (j <= 4)
			{
				table[j][index] = j;
				j++;
			}
		}
		else
		{
			index = index % 4;
			while (j <= 4)
			{
				table[j][index] = 5 - j;
				j++;
			}
		}

	}
}

void	find_4(char *str, char *table)
{
	int i;
	int j;

	i = 0;
	j = 0;
	while (*(str + i))
	{
		if (*(str + i) == '4')
		{
			record_1234(i);
		}
		i++;
	}
}

void	fill_table(char *str, char *table)
{
	find_4(str, table);



}

void	sudoku(char *str)
{
	int	i;
	int	cnt;
	int	depth;
	char	table[4][4];

	i = 0;
	depth = 0;
	cnt = 0;
	initialization(table);
	Erase_space(str);
	fill_table(str, table);
}

int main(int argc, char *argv[])
{
	sudoku(argv[1]);
	return 0;
}
