return {
	"catppuccin/nvim",
	name = "catppuccin",
	priority = 100,
	config = function()
		vim.g.catppuccin_flavour = "macchiato"
		vim.cmd("colorscheme catppuccin")
	end,
}
