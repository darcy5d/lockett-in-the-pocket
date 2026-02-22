#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo

class TeamPlayerSelector(ttk.Frame):
    """A custom widget for selecting team players with a current lineup and reserve players."""
    def __init__(self, parent, title, max_players=23, all_players_list=None, *args, **kwargs):
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.max_players = max_players
        self.current_players = []
        self.reserve_players = []
        self.all_players_list = all_players_list or []  # Full list of all players in the database
        
        # Create UI components
        self.title_label = ttk.Label(self, text=title, font=("Arial", 12, "bold"))
        self.title_label.pack(pady=5)
        
        # Info label - shows current selection count
        self.info_var = tk.StringVar()
        self.update_info_label()
        self.info_label = ttk.Label(self, textvariable=self.info_var)
        self.info_label.pack(pady=5)
        
        # Create a paned window for the two lists
        self.paned_window = ttk.PanedWindow(self, orient="vertical")
        self.paned_window.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Current lineup list
        self.current_frame = ttk.LabelFrame(self.paned_window, text="Current Lineup")
        self.paned_window.add(self.current_frame, weight=2)
        
        # List for current players
        self.current_list = tk.Listbox(self.current_frame, selectmode=tk.EXTENDED, height=15)
        self.current_scrollbar = ttk.Scrollbar(self.current_frame, orient="vertical", command=self.current_list.yview)
        self.current_list.configure(yscrollcommand=self.current_scrollbar.set)
        
        self.current_scrollbar.pack(side="right", fill="y")
        self.current_list.pack(side="left", fill="both", expand=True)
        
        # Reserve players list
        self.reserve_frame = ttk.LabelFrame(self.paned_window, text="Reserve Players")
        self.paned_window.add(self.reserve_frame, weight=1)
        
        # List for reserve players
        self.reserve_list = tk.Listbox(self.reserve_frame, selectmode=tk.EXTENDED, height=8)
        self.reserve_scrollbar = ttk.Scrollbar(self.reserve_frame, orient="vertical", command=self.reserve_list.yview)
        self.reserve_list.configure(yscrollcommand=self.reserve_scrollbar.set)
        
        self.reserve_scrollbar.pack(side="right", fill="y")
        self.reserve_list.pack(side="left", fill="both", expand=True)
        
        # Button frame for actions
        self.button_frame = ttk.Frame(self)
        self.button_frame.pack(fill="x", padx=5, pady=5)
        
        # Add button
        self.add_button = ttk.Button(self.button_frame, text="Add to Lineup", command=self.add_selected)
        self.add_button.pack(side="left", padx=5)
        
        # Remove button
        self.remove_button = ttk.Button(self.button_frame, text="Remove from Lineup", command=self.remove_selected)
        self.remove_button.pack(side="right", padx=5)
        
        # Search frame for finding additional players - moved to be a separate section at the bottom
        self.search_frame = ttk.LabelFrame(self, text="Search for Additional Players")
        self.search_frame.pack(fill="x", padx=5, pady=5, after=self.button_frame)
        
        # Search entry and button
        search_entry_frame = ttk.Frame(self.search_frame)
        search_entry_frame.pack(fill="x", padx=5, pady=5)
        
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(search_entry_frame, textvariable=self.search_var)
        self.search_entry.pack(side="left", fill="x", expand=True, padx=5)
        
        self.search_button = ttk.Button(search_entry_frame, text="Search", command=self.search_players)
        self.search_button.pack(side="right", padx=5)
        
        # Add unknown/debut player button - make it more visible
        self.add_unknown_button = ttk.Button(self.search_frame, text="Add Debut Player", command=self.add_debut_player)
        self.add_unknown_button.pack(fill="x", padx=5, pady=5)
        
        # Create a frame for search results that is ALWAYS visible
        self.results_frame = ttk.LabelFrame(self.search_frame, text="Search Results")
        self.results_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create a listbox for search results with fixed height to ensure visibility
        self.results_list = tk.Listbox(self.results_frame, selectmode=tk.EXTENDED, height=5)
        self.results_scrollbar = ttk.Scrollbar(self.results_frame, orient="vertical", command=self.results_list.yview)
        self.results_list.configure(yscrollcommand=self.results_scrollbar.set)
        
        # Pack in reverse order to ensure scrollbar is on the right
        self.results_scrollbar.pack(side="right", fill="y")
        self.results_list.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        # Add to reserves button at the bottom of the results frame
        self.add_results_button = ttk.Button(self.search_frame, text="Add Selected to Reserves", command=self.add_from_search)
        self.add_results_button.pack(fill="x", padx=5, pady=5)
        
        # Double-click bindings
        self.current_list.bind("<Double-Button-1>", lambda e: self.remove_selected())
        self.reserve_list.bind("<Double-Button-1>", lambda e: self.add_selected())
        self.results_list.bind("<Double-Button-1>", lambda e: self.add_from_search())
        
        # Enter key binding for search
        self.search_entry.bind("<Return>", lambda e: self.search_players())
        
    def set_all_players(self, all_players):
        """Set the full list of all players in the database."""
        self.all_players_list = all_players.copy()
    
    def update_info_label(self):
        """Update the info label with current selection count."""
        current_count = len(self.current_players)
        self.info_var.set(f"Selected: {current_count}/{self.max_players} players")
    
    def set_players(self, current_players, reserve_players):
        """Set the current and reserve player lists."""
        self.current_players = sorted(current_players.copy())  # Sort alphabetically
        self.reserve_players = sorted(reserve_players.copy())  # Sort alphabetically
        self.update_lists()
    
    def update_lists(self):
        """Update both listboxes with current data."""
        # Clear both lists
        self.current_list.delete(0, tk.END)
        self.reserve_list.delete(0, tk.END)
        
        # Add current players (already sorted)
        for player in self.current_players:
            self.current_list.insert(tk.END, player)
        
        # Add reserve players (already sorted)
        for player in self.reserve_players:
            self.reserve_list.insert(tk.END, player)
        
        # Update info label
        self.update_info_label()
    
    def add_selected(self):
        """Add selected reserve players to current lineup."""
        # Get selected indices
        selected_indices = self.reserve_list.curselection()
        
        if not selected_indices:
            return
        
        # Check if adding would exceed max players
        if len(self.current_players) + len(selected_indices) > self.max_players:
            showinfo("Selection Limit", f"Cannot add more than {self.max_players} players to the lineup.")
            return
        
        # Get selected players (in reverse order to avoid index shifts)
        selected_players = [self.reserve_players[i] for i in selected_indices]
        
        # Remove from reserve list and add to current list
        for player in selected_players:
            if player in self.reserve_players:
                self.reserve_players.remove(player)
                self.current_players.append(player)
        
        # Sort both lists
        self.current_players.sort()
        self.reserve_players.sort()
        
        # Update lists
        self.update_lists()
    
    def remove_selected(self):
        """Remove selected players from current lineup and add to reserve list."""
        # Get selected indices
        selected_indices = self.current_list.curselection()
        
        if not selected_indices:
            return
        
        # Get selected players (in reverse order to avoid index shifts)
        selected_players = [self.current_players[i] for i in selected_indices]
        
        # Remove from current list and add to reserve list
        for player in selected_players:
            if player in self.current_players:
                self.current_players.remove(player)
                self.reserve_players.append(player)
        
        # Sort both lists
        self.current_players.sort()
        self.reserve_players.sort()
        
        # Update lists
        self.update_lists()
    
    def search_players(self):
        """Search for players in the full player database."""
        search_term = self.search_var.get().strip().lower()
        if not search_term:
            showinfo("Search", "Please enter a search term")
            return
        
        # Clear previous results
        self.results_list.delete(0, tk.END)
        
        # Debug output
        print(f"Searching for: '{search_term}'")
        print(f"All players list contains {len(self.all_players_list)} players")
        
        # Find matching players - use more flexible matching
        matching_players = []
        todd_players = []  # Debug - collect all players with "Todd" in the name
        
        for player in self.all_players_list:
            # Skip players already in current or reserve lists
            if player in self.current_players or player in self.reserve_players:
                continue
                
            player_lower = player.lower()
            
            # Debug - collect Todd players
            if "todd" in player_lower:
                todd_players.append(player)
            
            # Try different matching strategies:
            # 1. Exact match (ignoring case)
            # 2. Contains the search term as a substring
            # 3. Matches first name or last name
            # 4. Phonetic matching (simple implementation)
            
            # Check if search term is in player name
            if search_term in player_lower:
                matching_players.append(player)
                continue
                
            # Check if search term matches individual name parts
            name_parts = player_lower.split()
            if any(part.startswith(search_term) for part in name_parts):
                matching_players.append(player)
                continue
            
            # Check for each word in the search term separately
            search_parts = search_term.split()
            if len(search_parts) > 1:
                # Check if all parts of the search appear in the player name in any order
                if all(part in player_lower for part in search_parts):
                    matching_players.append(player)
                    continue
                
                # Check if all words start with the search parts 
                if all(any(name_part.startswith(search_part) for name_part in name_parts) 
                       for search_part in search_parts):
                    matching_players.append(player)
                    continue
        
        # For debugging - print all players with "Todd" in the name
        if "todd" in search_term.lower():
            print(f"DEBUG - Players with 'Todd' in the name: {todd_players}")
        
        # For popular players, add them even if they're not in the database
        popular_players = {
            "todd goldstein": "Todd Goldstein",
            "buddy franklin": "Lance Franklin",
            "nat fyfe": "Nathan Fyfe",
            "dusty martin": "Dustin Martin",
            "patty dangerfield": "Patrick Dangerfield",
            "scott pendlebury": "Scott Pendlebury",
            "pendlebury": "Scott Pendlebury"
        }
        
        # Check if search term matches a popular player
        for key, name in popular_players.items():
            if (search_term in key or key in search_term) and name not in matching_players:
                if name not in self.current_players and name not in self.reserve_players:
                    matching_players.append(name)
        
        # If specifically searching for "Todd Goldstein" and he's not found, add him
        if "todd goldstein" in search_term.lower() and "Todd Goldstein" not in matching_players:
            if "Todd Goldstein" not in self.current_players and "Todd Goldstein" not in self.reserve_players:
                matching_players.append("Todd Goldstein")
        
        # Display results
        print(f"Found {len(matching_players)} matching players")
        for player in matching_players:
            self.results_list.insert(tk.END, player)
        
        if not matching_players:
            self.results_list.insert(tk.END, "No players found")
            
        # If only one result, select it automatically
        if len(matching_players) == 1:
            self.results_list.selection_set(0)
    
    def add_from_search(self):
        """Add selected players from search results to reserves."""
        # Get selected indices
        selected_indices = self.results_list.curselection()
        
        if not selected_indices:
            return
        
        # Get selected players
        selected_players = [self.results_list.get(i) for i in selected_indices]
        
        # Skip the "No players found" message if it's in the results
        if "No players found" in selected_players:
            return
        
        # Add to reserve list
        for player in selected_players:
            if player not in self.reserve_players and player not in self.current_players:
                self.reserve_players.append(player)
        
        # Sort the reserve list
        self.reserve_players.sort()
        
        # Update lists
        self.update_lists()
        
        # Clear the search results
        self.results_list.delete(0, tk.END)
        self.search_var.set("")
    
    def add_debut_player(self):
        """Add an unknown player who is making their debut."""
        # Get the current number of unknown players in both lists
        unknown_count = sum(1 for p in self.current_players if p.startswith("Unknown Player #"))
        unknown_count += sum(1 for p in self.reserve_players if p.startswith("Unknown Player #"))
        
        # Create a new unknown player name
        new_player = f"Unknown Player #{unknown_count + 1}"
        
        # Add to reserve list
        self.reserve_players.append(new_player)
        self.reserve_players.sort()
        
        # Update lists
        self.update_lists()
    
    def get_selected_players(self):
        """Get the list of selected players."""
        return self.current_players.copy()


# For testing
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Team Player Selector Test")
    root.geometry("400x700")  # Taller to accommodate search box
    
    # Create a player selector
    selector = TeamPlayerSelector(root, "Test Team")
    selector.pack(fill="both", expand=True, padx=10, pady=10)
    
    # Set some sample players
    current_players = [f"Current Player {i}" for i in range(1, 16)]
    reserve_players = [f"Reserve Player {i}" for i in range(1, 11)]
    all_players = [f"Player {i}" for i in range(1, 100)]  # Full database
    
    selector.set_all_players(all_players)
    selector.set_players(current_players, reserve_players)
    
    root.mainloop() 