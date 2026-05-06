import { Link } from "react-router";

export default function NavBar(){
    return (<nav className="flex gap-4 p-4 bg-gray-800">
        <Link to="/" className="text-white hover:underline">Papers</Link>
        <Link to="/search" className="text-white hover:underline">Search</Link>
    </nav>)
}